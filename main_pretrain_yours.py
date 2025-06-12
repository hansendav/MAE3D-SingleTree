# necessary imports torch, numpy, etc.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# model and data imports
from data_yours import FORAGE, FORSPECIES, ALS_50K
from model_yours import MAE3D
from util_yours import cal_loss_cd, write_plyfile


# wandb for metric logging and visualization 
import wandb


# hydra imports 
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate

# logging 
import logging



def _init_(cfg):
        # setup experiment dirs 
        experiment_dir = Path(f'./experiments/{cfg.experiment_setup.name}')
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        for i in ['checkpoints', 'visualization', 'logs']:
            (experiment_dir / i).mkdir(parents=True, exist_ok=True)


        logger = logging.getLogger(__name__) 

        logger.info(f'Experiment initalized: {cfg.experiment_setup.name}, mode: {cfg.mode}')

        # paths to config 
        cfg.experiment_setup.experiment_dir = experiment_dir
        cfg.experiment_setup.checkpoints_dir = experiment_dir / 'checkpoints'
        cfg.experiment_setup.visualization_dir = experiment_dir / 'visualization'
        cfg.experiment_setup.logs_dir = experiment_dir / 'logs'

        return logger


def _set_reproducability(seed, deterministic=True):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training(cfg, logger): 
    # setup wandb for loggin metrics 
    wandb.init(
        project=f'MAE3D-SingleTree', 
        name=f'{cfg.experiment_setup.name}_{cfg.mode}',
        mode='online'
    )


    # setup datasets 

    if cfg.pretraining.dataset.name == 'ALS_50K': 
         train_loader = DataLoader(
              ALS_50K(
                   num_points=cfg.model.num_points,
                   # to-do: augmentations 
              ), 
              batch_size=cfg.pretraining.batch_size,
              shuffle=cfg.pretraining.shuffle,
              num_workers=cfg.pretraining.num_workers,
              pin_memory=cfg.pretraining.pin_memory,
              drop_last=True # to ensure all batches have the same size

         )

    # set device 
    device = torch.device("cuda" if cfg.experiment_setup.use_cuda else "cpu")

    # setup model 
    model = MAE3D(cfg.model).to(device) 
    logger.info(f'Model initialized with cfg: {cfg.model}')

    # setup optimizer
    optimizer = instantiate(
         cfg.pretraining.optimizer, 
         params=model.parameters() 
    )
    logger.info(f'Instantiated optimizer: {cfg.pretraining.optimizer}')

    # setup scheduler 
    scheduler = instantiate( 
         cfg.pretraining.scheduler,
         optimizer=optimizer
    )
    logger.info(f'Instantiated scheduler: {cfg.pretraining.scheduler}')

    # resume training if specified
    if cfg.pretraining.resume:
        model.load_state_dict(torch.load(cfg.pretraining.resume_path))
        logger.info(f'Loaded model from: {cfg.pretraining.resume_path}')

    # Pretraining loop 
    for epoch in range(cfg.pretraining.epochs):
        train_center_loss = 0.0
        train_pc_loss = 0.0 
        train_loss = 0.0
        chamfer_dist = 0.0
        count = 0.0
        model.train()

        idx = 0
        for batch, (data, index) in enumerate(tqdm(train_loader)):
            data = data.float()
            index = index.long()
            data = data.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            optimizer.zero_grad()

            pred_pc, pred_center, gt_center, vis_pos, crop_pos = model(data)

            loss_cd_center = cal_loss_cd(pred_center, gt_center.permute(0, 2, 1))
            loss_cd_pc = cal_loss_cd(pred_pc, data) # data shape [B, 3, N]
            chamfer_dist_pc = cal_loss_cd(pred_pc, data, mode='mean')

            loss = loss_cd_pc * cfg.model.loss_alpha_center + loss_cd_center
            loss.backward()
            optimizer.step()

            # instance count and loss accumulation
            count += batch_size
            train_loss += loss.item() * batch_size
            train_center_loss += loss_cd_center.item() * batch_size
            train_pc_loss += loss_cd_pc.item() * batch_size
            chamfer_dist += chamfer_dist_pc.item() * batch_size

            # increment step index
            idx += 1

            # write to wandb logger per step
            wandb.log({
                'step': idx,
                'step_loss': loss.item(),
                'step_center_loss': loss_cd_center.item(),
                'step_pc_loss': loss_cd_pc.item(),
                'step_cd_pc': chamfer_dist_pc.item(),        
            })
            
            # Visualization of last batch of last epoch
            if epoch == cfg.pretraining.epochs - 1 and cfg.experiment_setup.visualize:
                for i in range(batch_size):
                    write_plyfile(str(cfg.experiment_setup.visualization_dir / f'train_{index[i].item()}_vis'), vis_pos[i].view(-1, 3))
                    write_plyfile(str(cfg.experiment_setup.visualization_dir / f'train_{index[i].item()}_mask'), crop_pos[i].view(-1, 3))
                    write_plyfile(str(cfg.experiment_setup.visualization_dir / f'train_{index[i].item()}_gt'), data[i].permute(1, 0))
                    write_plyfile(str(cfg.experiment_setup.visualization_dir / f'train_{index[i].item()}_pred'), pred_pc[i].permute(1, 0))

        # per epoch 
        scheduler.step()

        train_loss /= count 
        train_center_loss /= count
        train_pc_loss /= count
        chamfer_dist /= count

        logger.info( 
            f'Epoch: {epoch}, \n'
            f'Train Loss: {train_loss:.6f}, \n'
            f'Train Center Loss: {train_center_loss:.6f}, \n'
            f'Train PC Loss: {train_pc_loss:.6f}, \n'
            f'Chamfer Distance: {chamfer_dist:.6f}'
        )

        # write epoch metrics to wandb
        wandb.log({
            'epoch': epoch, 
            'epoch_loss': train_loss,
            'epoch_center_loss': train_center_loss,
            'epoch_pc_loss': train_pc_loss,
            'epoch_cd': chamfer_dist,
        })

        if epoch % cfg.pretraining.save_interval == 0:
            torch.save(model.state_dict(), str(cfg.experiment_setup.checkpoints_dir / f'pretrained_epoch_{epoch}.pth'))
            logger.info(f'Model saved at epoch {epoch}.')
        if epoch == cfg.pretraining.epochs - 1:
            torch.save(model.state_dict(), str(cfg.experiment_setup.checkpoints_dir / f'pretrained.pth'))
            logger.info(f'Last model saved {epoch}.')

    wandb.finish()

@hydra.main(config_path="config/experiments", version_base=None)
def main(cfg: DictConfig):
    logger = _init_(cfg)

    # print experiment config -> can also be seen in ./experiment_name/.hydra/config.yaml
    logger.info(f'Experiment config: {cfg}')
    logger.info(f'Pretraining dataset config: {cfg.pretraining.dataset.name}')

    # setup reproducability: default seed 42, deterministic=True, benchmark=False
    _set_reproducability(cfg.experiment_setup.seed, cfg.experiment_setup.deterministic)

    # check if CUDA is available
    if cfg.experiment_setup.use_cuda:
        if torch.cuda.is_available():
            logger.info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
        else: 
            logger.warning('CUDA is not available, using CPU instead.')
            cfg.pretraining.use_cuda = False


    # pretraining with wandb logging
    training(cfg, logger)

if __name__ == "__main__":
    main()
    