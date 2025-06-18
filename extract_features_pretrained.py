import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from data_yours import FORAGE, FORSPECIES, ALS_50K
from model_yours import MAE3D

# hydra imports 
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate

# logging 
import logging

# pandas 
import pandas as pd 


def _init_(cfg):

    experiment_dir = Path(f'./experiments/{cfg.experiment_setup.name}')
    cfg.experiment_dir = experiment_dir
      
    logger = logging.getLogger(__name__) 

    # paths to config 
    cfg.experiment_setup.experiment_dir = experiment_dir
    cfg.experiment_setup.checkpoints_dir = experiment_dir / 'checkpoints'
    cfg.experiment_setup.visualization_dir = experiment_dir / 'visualization'
    cfg.experiment_setup.logs_dir = experiment_dir / 'logs'
    # check if necessary directories exists and files exists 
    if not Path(cfg.experiment_setup.experiment_dir).exists():
        raise FileNotFoundError(f"Experiment directory '{cfg.experiment_setup.experiment_dir}' does not exist.")

    if not Path(f'{cfg.experiment_setup.checkpoints_dir}/pretrained.pth').exists():
        raise FileNotFoundError(f"Pretrained model file '{cfg.experiment_setup.checkpoints_dir}/pretrained.pth' does not exist.")

    Path(cfg.experiment_setup.experiment_dir / 'feature_embeddings').mkdir(parents=True, exist_ok=True)

    logger.info(f'Pretrained model found. Dir for embeddings created at: {str(Path(cfg.experiment_setup.experiment_dir / 'feature_embeddings'))}')

    # paths to config 
    return logger


def _set_reproducability(seed, deterministic=True):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@hydra.main(config_path="config/experiments", version_base=None)
def main(cfg: DictConfig):

    # init for feature embedding creation
    logger = _init_(cfg)

    # set up reproducibility
    _set_reproducability(cfg.experiment_setup.seed)

    # set device 
    device = torch.device("cuda" if cfg.experiment_setup.use_cuda else "cpu")


    # load the pretrained model and load checkpoint 
    model = MAE3D(cfg.model) 
    checkpoint = torch.load(cfg.experiment_setup.checkpoints_dir / 'pretrained.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # select only the patch embedding layer 
    emb_module = model.encoder.patch_embed
    encoder = model.encoder

    logger.info(f'Loaded pretrained model from {cfg.experiment_setup.checkpoints_dir / "pretrained.pth"}    ')

    # set both to eval and move to device 
    emb_module.eval()
    emb_module.to(device)
    encoder.eval()
    encoder.to(device)

    # get the dataset 
    if cfg.finetune.dataset.name == 'FORAGE': 
        process_loader = DataLoader(
            FORAGE(),
            batch_size=cfg.finetune.dataset.batch_size,
            shuffle=cfg.finetune.shuffle,
            num_workers=cfg.finetune.num_workers,
            pin_memory=cfg.finetune.pin_memory,
            drop_last=False,
            prefetch_factor=cfg.finetune.prefetch_factor
        )
        logger.info(f"Process loader created with {len(process_loader)} batches.")

    # setup save path for embeddings 
    save_path = Path(cfg.experiment_setup.experiment_dir) / 'feature_embeddings') 
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Embeddings will be saved to {save_path}")

    # lists to store embeddings and labels 
    emb_patch_module = [] 
    emb_encoder = [] 
    labels = [] 

    with torch.no_grad(): 
        for batch in tqdm(process_loader, desc="Processing batches"):
            data, labels = batch
            
            data = data.to(device) 
        

            # get the embeddings from the patch embedding module 
            emb_patch = emb_module(data)
            emb_patch_module.append(emb_patch.cpu().numpy())

            # get the embeddings from the encoder 
            emb_enc = encoder(data)
            emb_encoder.append(emb_enc.cpu().numpy())

            # get the labels 
            labels.append(batch.y.cpu().numpy())

    # convert lists to numpy arrays
    emb_patch_module = np.concatenate(emb_patch_module, axis=0)
    emb_encoder = np.concatenate(emb_encoder, axis=0)
    labels = np.concatenate(labels, axis=0)

    logger.info(f"Embeddings shape: {emb_patch_module.shape}, {emb_encoder.shape}, Labels shape: {labels.shape}")

    # create dataframes to save embeddigns and labels to csv 
    df_patch = pd.DataFrame(emb_patch_module)
    df_encoder = pd.DataFrame(emb_encoder)

    df_patch['age'] = labels[:, 0]
    df_encoder['age'] = labels[:, 0] 

    # write to csv 
    df_patch.to_csv(save_path / 'embs_patch.csv', index=False)
    df_encoder.to_csv(save_path / 'embs_encoder.csv', index=False)

    logger.info(f"Embeddings saved to {save_path / 'embs_patch.csv'} and {save_path / 'embs_encoder.csv'}")
    logger.info("Feature extraction completed successfully.")

if __name__ == "__main__":
    main()