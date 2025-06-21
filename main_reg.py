# necessary imports torch, numpy, etc.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# model and data imports
from data_yours import FORAGE, FORSPECIES
from model_yours import MAE3D, MAE3D_reg, MAE3D_cls

# metrics
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError

# wandb for metric logging and visualization 
import wandb


# hydra imports 
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate

# logging 
import logging

# dataframes 
import pandas as pd


def _init_(cfg):
        # setup experiment dirs 
        experiment_dir = Path(f'./experiments/{cfg.experiment_setup.name}')
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        for i in ['checkpoints', 'visualization', 'logs', 'results']:
            (experiment_dir / i).mkdir(parents=True, exist_ok=True)


        logger = logging.getLogger(__name__) 

        logger.info(f'Experiment initalized: {cfg.experiment_setup.name}, mode: {cfg.mode}')

        # paths to config 
        cfg.experiment_setup.experiment_dir = experiment_dir
        cfg.experiment_setup.checkpoints_dir = experiment_dir / 'checkpoints'
        cfg.experiment_setup.visualization_dir = experiment_dir / 'visualization'
        cfg.experiment_setup.logs_dir = experiment_dir / 'logs'
        cfg.experiment_setup.results_dir = experiment_dir / 'results'

        return logger


def _set_reproducability(seed, deterministic=True):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training(cfg, logger):

    # setup datasets no other datasets are supported yet
    if cfg.downstream.dataset.name == 'FORAGE':
        train_loader = DataLoader(
            FORAGE(
                split='train',
                fraction=cfg.downstream.dataset.fraction),
                num_workers=cfg.downstream.num_workers, 
                batch_size=cfg.downstream.batch_size, 
                shuffle=cfg.downstream.shuffle, 
                drop_last=True,
                pin_memory=cfg.downstream.pin_memory,
                prefetch_factor=cfg.downstream.prefetch_factor
            )

        val_loader = DataLoader(
            FORAGE(
                split='val',
                fraction=1), # always use full validation set
                num_workers=cfg.downstream.num_workers, 
                batch_size=cfg.downstream.batch_size, 
                shuffle=False, # no need to shuffle validation set
                drop_last=False, # no drop of last for validation
                pin_memory=cfg.downstream.pin_memory,
                prefetch_factor=cfg.downstream.prefetch_factor
            )

        test_loader = DataLoader(
            FORAGE(
                split='test',
                fraction=1), # always use full test set
                num_workers=cfg.downstream.num_workers, 
                batch_size=cfg.downstream.batch_size, 
                shuffle=False, 
                drop_last=False,
                pin_memory=cfg.downstream.pin_memory,
                prefetch_factor=cfg.downstream.prefetch_factor
            )

    # log dataset and fraction information 
    logger.info(
        f'Using dataset: {cfg.downstream.dataset}')
    
    logger.info(f'Resulting in train set size: {len(train_loader.dataset)}')
    logger.info(f'Resulting in validation set size: {len(val_loader.dataset)}')
    logger.info(f'Resulting in test set size: {len(test_loader.dataset)}')

    # set device
    device = torch.device("cuda" if cfg.experiment_setup.use_cuda else "cpu")

    # set repetition lists 
    val_r2_reps = []
    val_rmse_reps = []
    test_r2_reps = []
    test_rmse_reps = []
    
    # FOR EACH REPETITION:
    for rep in range(cfg.downstream.repetitions):

        # setup new wandb logger 
        # setup wandb for loggin metrics 
        wandb.init(
            project=f'MAE3D-SingleTree', 
            name=f'{cfg.experiment_setup.name}_{cfg.mode}_{cfg.downstream.dataset.fraction}_{rep+1}',
            mode='online' # online streaming of metrics
        )
        
        if cfg.downstream.task == 'regression':
            model = MAE3D_reg(cfg.model).to(device)
            logger.info(f'Regression model initialized with cfg: {cfg.model}')
        elif cfg.downstream.task == 'classification':
            model = MAE3D_cls(cfg.model).to(device)
            logger.info(f'Classification model initialized with cfg: {cfg.model}')
        

        # CHANGED loading 
        if cfg.downstream.pretrained: 
            logger.info(
                f'Loading pretrained model from:\n'
                f'{cfg.experiment_setup.checkpoints_dir}/pretrained.pth'
                )
            model_pretrain = MAE3D(cfg.model).to(device)

            if cfg.pretraining.dataset.name == 'ALS_50K':
                pretrained_dict = torch.load(f'{cfg.experiment_setup.checkpoints_dir}/pretrained.pth', map_location=device)
                model_checkpoint = pretrained_dict['model_state_dict']
            elif cfg.pretraining.dataset.name == 'FORSPECIES':
                model_checkpoint = torch.load(f'{cfg.experiment_setup.checkpoints_dir}/pretrained.pth', map_location=device)

            # extract weights from patch_embed 
            point_embed_dict = {k: v for k, v in model_checkpoint.items() if 'patch_embed' in k}

            model_dict = model.state_dict() 

            # set weights in the new model dict
            for k, v in point_embed_dict.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)

            logger.info(f'Pretrained patch embedding module loaded.')
        else: 
            logger.info(f'Pretrained weights in patch embedding module not loaded.')

        # freeze patch embedding module if linear classifier 
        if cfg.downstream.train_type == 'probing':
            for n, p in model.named_parameters():
                if n.split('.')[1] == 'patch_embed':
                    p.requires_grad = False
            logger.info(f'Linear probing mode. Updating only regression head weights.')
        # full fine-tuning setup
        elif cfg.downstream.train_type == 'finetune': 
            assert cfg.downstream.pretrained is True, f'Fine-tuning requires pretrained to be True.'
            logger.info(f'Full fine-tuning mode. Updating all parameters (embeding module and regression head).')
        
        # from scratch training setup
        elif cfg.downstream.train_type == 'from_scratch': 
            assert cfg.downstream.pretrained is False, f'From scratch training requires pretrained to be False.'
            logger.cprint(f'Full training from scratch. Updating all parameters (embedding and regressor head). No pretrained weights loaded.')

        # setup optimizer, schedulr and criterion
        optimizer = instantiate(
            cfg.downstream.optimizer, 
            params=model.parameters() 
        )
        logger.info(f'Instantiated optimizer: {cfg.pretraining.optimizer}')

        # setup scheduler 
        schedulers = [instantiate(scheduler, optimizer=optimizer) for scheduler in cfg.downstream.schedulers]
        scheduler = SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=[schedulers[0].total_iters] # assumin the first scheduler is LinearLR 
        )

        logger.info(f'Instantiated schedulers: {cfg.downstream.schedulers}')

        # resume training if specified
        if cfg.downstream.resume:
            checkpoint = torch.load(cfg.downstream.resume_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 
            logger.info(f'Loaded model and optimizer from: {cfg.pretraining.resume_path}')
            logger.info(f'Resuming training from epoch {start_epoch}.')
        else: 
            start_epoch = 0
        
        # criterion
        if cfg.downstream.task == 'regression':
            criterion = instantiate(cfg.downstream.regression_criterion)

        logger.info(f'Using criterion: {criterion}')
        
        # set up torchmetrics, bring to device 
        r2_train = R2Score().to(device)
        mse_train = MeanSquaredError().to(device)
        mae_train = MeanAbsoluteError().to(device)

        r2_val = R2Score().to(device)
        mse_val = MeanSquaredError().to(device)
        mae_val = MeanAbsoluteError().to(device)
    
    


        i = rep + 1
        logger.info(f'Running repetition {i} of {cfg.downstream.repetitions}.')

        # best rmse, r2 to save best model
        best_val_rmse =  1e10 # best model has to be <
        best_val_r2 = -1e10 # best model has to be > 

        # reset loss for each repetition 
        train_loss = 0.0
        val_loss = 0.0
        count = 0.0

        for epoch in range(cfg.downstream.epochs):
            train_loss = 0.0
            count = 0.0
            model.train()

            # reset train metrics for each epoch
            r2_train.reset() 
            mse_train.reset() 
            mae_train.reset()

            for batch, (data, age, species) in enumerate(tqdm(train_loader)):
                data, age = data.to(device).float(), age.squeeze(-1).to(device)
                data = data.permute(0, 2, 1) 
                batch_size = data.size()[0]

                optimizer.zero_grad()

                preds = model(data)

                loss = criterion(preds.squeeze(-1), age)
                loss.backward()


                # clip gradients if specified
                if cfg.downstream.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.downstream.clip_grad_norm)

                if cfg.downstream.clip_grad_value is not None:
                    nn.utils.clip_grad_value_(model.parameters(), cfg.downstream.clip_grad_value)

                optimizer.step()
                
                count += batch_size
                train_loss += loss.item() * batch_size
                
                r2_train.update(preds.squeeze(-1), age) 
                mse_train.update(preds.squeeze(-1), age)
                mae_train.update(preds.squeeze(-1), age)


            # scheduler step after each epoch
            wandb.log({
                'lr': optimizer.param_groups[0]['lr'],
            })
            scheduler.step()
            
            epoch_r2 = r2_train.compute().detach().cpu().item()
            epoch_mse = mse_train.compute()
            epoch_rmse = torch.sqrt(epoch_mse).detach().cpu().item()
            epoch_mae = mae_train.compute().detach().cpu().item()
            

            logger.info(f'Train {epoch}, loss: {train_loss * 1.0 / count}, train rmse: {epoch_rmse}, train r2: {epoch_r2}, train mae: {epoch_mae}')
            
            wandb.log({
                'epoch': epoch, 
                'train/epoch_loss': train_loss * 1.0 / count,
                'train/epoch_rmse': epoch_rmse,
                'train/epoch_r2': epoch_r2,
                'train/epoch_mae': epoch_mae,
            })

            ### VALIDATION ###
            # validate every 10 epochs, save best model every 10 epochs
            if epoch % 10 == 0:

                logger.info(f'Evaluation on validation set at epoch {epoch}')
                
                val_loss = 0.0
                count = 0.0
                total_time = 0.0


                # set model in evaluation mode 
                model.eval() 

                mae_val.reset() 
                mse_val.reset() 
                r2_val.reset()

                # list for savind predictions and gt
                gt = []
                predictions = []
                species_list = []

                with torch.no_grad():
                    for data, age, species in tqdm(val_loader):
                        data, age = data.to(device).float(), age.squeeze(-1).to(device)
                        data = data.permute(0, 2, 1)
                        batch_size = data.size()[0]

                        preds = model(data)
                        loss = criterion(preds.squeeze(-1), age)
                        
                        count += batch_size
                        val_loss += loss.item() * batch_size

                        r2_val.update(preds.squeeze(-1), age)
                        mse_val.update(preds.squeeze(-1), age)
                        mae_val.update(preds.squeeze(-1), age)

                        # append predictions and gt to list 
                        gt.append(age)
                        predictions.append(preds)
                        species_list.append(species)                    

                    epoch_val_r2 = r2_val.compute().detach().cpu().item()
                    epoch_val_mse = mse_val.compute()
                    epoch_val_rmse = torch.sqrt(epoch_val_mse).detach().cpu().item()
                    epoch_val_mae = mae_val.compute().detach().cpu().item()
                    epoch_val_mbe = torch.mean(torch.cat(predictions) - torch.cat(gt)).detach().cpu().item()

                    r2_val.reset()
                    mse_val.reset()
                    mae_val.reset()


                    wandb.log({
                        'val/epoch_loss': val_loss * 1.0 / count,
                        'val/epoch_rmse': epoch_val_rmse,
                        'val/epoch_r2': epoch_val_r2,
                        'val/epoch_mae': epoch_val_mae,
                        'val/epoch_mbe': epoch_val_mbe,
                        'val/epoch_mse': epoch_val_mse,
                        'epoch': epoch
                    })

                    logger.info(f'Validation {epoch}, loss: {val_loss * 1.0 / count:.4f}, val rmse: {epoch_val_rmse:.4f}, val r2: {epoch_val_r2:.4f}')

                    if epoch_val_r2 >= best_val_r2:
                        best_val_r2 = epoch_val_r2
                        best_val_rmse = epoch_val_rmse

                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch
                        }
                        if cfg.downstream.train_type == 'probing': 
                            torch.save(checkpoint, Path(f'{cfg.experiment_setup.checkpoints_dir}/model_reg_probing_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'))
                        elif cfg.downstream.train_type == 'finetune': 
                            torch.save(checkpoint, Path(f'{cfg.experiment_setup.checkpoints_dir}/model_reg_ft_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'))
                        else: 
                            torch.save(checkpoint, Path(f'{cfg.experiment_setup.checkpoints_dir}/model_reg_scratch_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'))
                        
                        logger.info(f'Best model RMSE: {best_val_rmse}, R2Score:{best_val_r2}')


                        # Save predictions
                        gt = torch.cat(gt, dim=0).cpu().numpy()
                        predictions = torch.cat(predictions, dim=0).cpu().numpy()
                        species_list = torch.cat(species_list, dim=0).cpu().numpy()

                        validation_results = pd.DataFrame({
                            'Ground Truth': gt.flatten(),
                            'Predictions': predictions.flatten(),
                            'Species': species_list.flatten()
                        })

                        # Save validation predictions
                        validation_results_file_name = f'validation_results_fraction_{cfg.downstream.dataset.fraction}_{rep}.csv'
                        validation_results.to_csv(
                            cfg.experiment_setup.results_dir / validation_results_file_name, index=False)
                        logger.info(f'Validation results saved to {validation_results_file_name}')

        val_r2_reps.append(best_val_r2)
        val_rmse_reps.append(best_val_rmse)
        logger.info(f'Best validation RMSE: {best_val_rmse}, R2Score:{best_val_r2}')

        ### TESTING ###
        logger.info(f'Evaluating on test set:')

        # reset all metrics
        mae_val.reset()
        mse_val.reset()
        r2_val.reset()

        # load the best model (see validation)
        if cfg.downstream.train_type == 'finetune':
            checkpoint_path = cfg.experiment_setup.checkpoints_dir / f'model_reg_ft_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'        
        elif cfg.downstream.train_type == 'probing':
            checkpoint_path = cfg.experiment_setup.checkpoints_dir / f'model_reg_probing_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'
        else:
            checkpoint_path = cfg.experiment_setup.checkpoints_dir / f'model_reg_scratch_limited_{cfg.downstream.dataset.fraction}_{rep}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded best model from {checkpoint_path}')
        

        # evaluation
        model.eval()
        gt = []
        predictions = []
        species_list = []

        with torch.no_grad():
            for batch, (data, age, species) in enumerate(tqdm(test_loader)):
                data, age = data.to(device).float(), age.squeeze(-1).to(device)
                data = data.permute(0, 2, 1)
                
                preds = model(data)

                # update metrics 
                r2_val.update(preds.squeeze(-1), age)
                mse_val.update(preds.squeeze(-1), age)
                mae_val.update(preds.squeeze(-1), age)

                gt.append(age)
                predictions.append(preds)
                species_list.append(species)

            # Concatenate all batches
            gt = torch.cat(gt, dim=0).cpu().numpy()
            predictions = torch.cat(predictions, dim=0).cpu().numpy()
            species_list = torch.cat(species_list, dim=0).cpu().numpy()

            # Calculate metrics
            mae = mae_val.compute().detach().cpu().item()
            mse = mse_val.compute().detach().cpu().item()
            r2 = r2_val.compute().detach().cpu().item()
            rmse = np.sqrt(mse)
            mbe = np.mean(predictions - gt)

            logger.info(
            f'TEST METRICS: \n'
            f'MAE: {mae:.2f}, \n'
            f'MSE: {mse:.2f}, \n'
            f'RMSE: {rmse:.2f}, \n'
            f'R2: {r2:.2f}, \n'
            f'MBE: {mbe:.2f}, \n'
            )

            # Log test metrics to wandb
            wandb.log({
            'test/mae': mae,
            'test/mse': mse,
            'test/rmse': rmse,
            'test/r2': r2,
            'test/mbe': mbe,
            })

            # Save predictions
            test_results = pd.DataFrame({
            'gt': gt.flatten(),
            'pred': predictions.flatten(),
            'species': species_list.flatten()
            })

            test_results_file_name = f'test_results_{cfg.downstream.train_type}_{cfg.downstream.dataset.fraction}_{rep}.csv'
            test_results.to_csv(cfg.experiment_setup.results_dir / test_results_file_name, index=False)
            logger.info(f'Test results saved to {str(cfg.experiment_setup.results_dir / test_results_file_name)}')   

            test_r2_reps.append(r2)
            test_rmse_reps.append(rmse)

            logger.info(f'Best test RMSE: {rmse}, R2Score:{r2}')

        # end of repetitions loop
        wandb.finish()

    # Save all metrics for all repetitions  
    metrics_df = pd.DataFrame({
        'Validation RMSE': val_rmse_reps,
        'Validation R2': val_r2_reps,
        'Test RMSE': test_rmse_reps,
        'Test R2': test_r2_reps
    })

    # Calculate mean and std for each metric
    mean_row = metrics_df.mean().round(2)
    std_row = metrics_df.std().round(2)

    # Append mean and std as new rows
    metrics_df  = pd.concat([metrics_df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)

    # Set index names for mean and std
    metrics_df.index = [f'rep_{i+1}' for i in range(len(val_rmse_reps))] + ['mean', 'std']

    # Save to CSV
    metrics_save_path = cfg.experiment_setup.results_dir / f'{cfg.downstream.train_type}_{cfg.downstream.dataset.fraction}_allrepetitions.csv'
    metrics_df.to_csv(metrics_save_path)
    logger.info(f'Experiment metrics saved to {str(metrics_save_path)}')
  

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