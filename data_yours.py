#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.model_selection import train_test_split
#import laspy

import fpsample


def normalize_pointcloud(pointcloud: np.ndarray) -> np.ndarray: 

    # assert 
    assert pointcloud.shape[1] == 3, "Pointcloud should be of shape (N, 3)"

    # center pointcloud into origin (z-axis too)
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    
    # farthest distance to origin 
    # normalize to unit sphere 
    m = np.max(np.sqrt(np.sum(pointcloud ** 2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud.astype(np.float32)


def translate_pointcloud(pointcloud, scale_range=[0.6, 1.5], translation_range=[-0.01, 0.01]):
    
    if isinstance(pointcloud, torch.Tensor):
        scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
        translation = torch.empty(1).uniform_(translation_range[0], translation_range[1]).item()

        translated_pointcloud = (pointcloud * scale + translation).float()
        return translated_pointcloud

    else:
        scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
        translation = np.random.uniform(low=translation_range[0], high=translation_range[1]) 

        translated_pointcloud = np.add(pointcloud*scale, translation).astype('float32')
        return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):

    if isinstance(pointcloud, torch.Tensor):
        N, C = pointcloud.shape
        pointcloud += torch.clamp(sigma * torch.randn(N, C), min=-1 * clip, max=clip)
        return pointcloud
    else:
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return pointcloud

def flip_pointcloud(pointcloud, p=0.5): 
    if np.random.rand() < p: 
        pointcloud[:, :2] = -pointcloud[:, :2]

    return pointcloud

def rotate_pointcloud(pointcloud):

    if isinstance(pointcloud, torch.Tensor):
        theta = torch.empty(1).uniform_(0, 2 * torch.pi)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
            ]
        )
        pointcloud = torch.matmul(pointcloud, rotation_matrix.T)
        return pointcloud
    else:
        theta = np.pi * 2 * np.random.uniform()
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]

            ]
        )
        pointcloud = pointcloud.dot(rotation_matrix)  # random rotation z-axis
        
        return pointcloud

def subsample_pointcloud(pointcloud: np.ndarray, num_points: int, random: bool = True) -> np.ndarray:
    """
    Subsamples the pointcloud to the specified number of points.

    Args:
        pointcloud (np.ndarray): Input pointcloud of shape (N, 3).
        num_points (int): Number of points to subsample to.
        random (bool): Whether to randomly sample points. If False perform Furthest Point Sampling.

    Returns:
        np.ndarray: Subsampled pointcloud of shape (num_points, 3).
    """
    assert pointcloud.shape[1] == 3, "Pointcloud should be of shape (N, 3)"
    assert pointcloud.shape[0] > num_points, "Pointcloud should have at least num_points points"

    if random:
        indices = np.random.choice(pointcloud.shape[0], num_points, replace=False)
        pointcloud = pointcloud[indices]
    else:
        pointcloud_t = torch.tensor(pointcloud[None, :], dtype=torch.float32).cuda()
        fps_idxs = pointnet2_utils.furthest_point_sample(pointcloud_t, num_points) 
        fps_idxs = fps_idxs.cpu().numpy()
        fps_idxs = np.squeeze(fps_idxs, axis=0)
        pointcloud = pointcloud[fps_idxs]
    
    return pointcloud



def add_points(pointcloud: np.ndarray, num_points: int) -> np.ndarray: 
    """
    Repeats random points in pointcloud to match num_points. 
    """

    assert pointcloud.shape[1] == 3, "Pointcloud should be of shape (N, 3)"
    assert pointcloud.shape[0] <= num_points, "Pointcloud should have less points than num_points"

    point_diff = num_points - pointcloud.shape[0]

    # select random points from pointcloud
    random_points = np.random.choice(pointcloud.shape[0], point_diff, replace=True)

    pointcloud = np.concatenate((pointcloud, pointcloud[random_points]), axis=0)
    return pointcloud


class FORSpecies(Dataset): 
    def __init__(self, num_points, split='train', pretrain=True):
        self.split = split 
        self.data_path = '/home/nibio/mutable-outside-world/data/forspecies/dev'
        
        self.ds_file = '/home/nibio/mutable-outside-world/data/forspecies/tree_metadata_dev.csv'
        self.ds_meta = pd.read_csv(self.ds_file) 

        # split dataset into train and val 
        train_df, val_df = train_test_split(self.ds_meta, test_size=0.2, stratify=self.ds_meta['species'], random_state=42)

        if self.split == 'train': 
            self.data_list = train_df['filename'].map(lambda x: x.split('/')[-1].replace('.las', '.laz')).to_list()
            del val_df

        elif self.split == 'val':
            self.data_list = val_df['filename'].map(lambda x: x.split('/')[-1].replace('.las', '.laz')).to_list()
            del train_df
        
        self.index = np.arange(len(self.data_list))
        self.num_points = num_points 

        self.pretrain = pretrain

    def __getitem__(self, idx): 

        if self.pretrain == True:
            las = laspy.read(os.path.join(self.data_path, self.data_list[idx]))
            pointcloud = np.vstack((las.x, las.y, las.z)).astype(np.float64).T
            index = self.index[idx]

            pointcloud = normalize_pointcloud(pointcloud)

            if self.split == 'train':
                pointcloud = flip_pointcloud(pointcloud)
                pointcloud = translate_pointcloud(pointcloud)
                pointcloud = rotate_pointcloud(pointcloud)
                pointcloud = jitter_pointcloud(pointcloud)

            # constrain pointcloud to num_points
            if pointcloud.shape[0] > self.num_points:
                pointcloud = subsample_pointcloud(pointcloud, self.num_points, random=True)
            elif pointcloud.shape[0] < self.num_points:
                pointcloud = add_points(pointcloud, self.num_points)
            
            return torch.tensor(pointcloud, dtype=torch.float32), index

    def __len__(self):
        return len(self.data_list)

class FORSPECIES(Dataset): 
    def __init__(self, split='train', mode='pretrain'):
        super().__init__()
        self.split = split 
        self.mode = mode
        self.data_file = Path('/home/nibio/mutable-outside-world/code/david_hansen/ssl_single_tree_pool/SingleTreeDB/data/datasets/FORSpecies/FORSpecies_1.0.h5')
        
        if self.mode == 'pretrain':
            with h5py.File(self.data_file, 'r') as f:
                self.len = f[self.split]['clouds'].shape[0]


    def __len__(self):
        return self.len 

    def __getitem__(self, idx): 
        with h5py.File(self.data_file, 'r', swmr=True) as f: 
            cloud = f[self.split]['clouds'][idx]

        
        cloud = normalize_pointcloud(cloud)
        cloud = translate_pointcloud(cloud) 
        cloud = jitter_pointcloud(cloud)
        cloud = rotate_pointcloud(cloud)
        cloud = flip_pointcloud(cloud)


        return cloud, idx
    


class FORAGE(Dataset): 
    def __init__(self, split='train', fraction=1.0):
        super().__init__()
        self.split = split 
        self.data_file = Path('/home/nibio/mutable-outside-world/code/david_hansen/ssl_single_tree_pool/SingleTreeDB/data/datasets/FORAge/FORAge_1.0.h5')
        
        
        with h5py.File(self.data_file, 'r') as f:
            self.len = f[self.split].attrs['num_instances']

        self.fraction = fraction 
        
        if self.split == 'train' and self.fraction < 1.0:
            with h5py.File(self.data_file, 'r') as f:
                age = f[self.split]['age'][:]
                selection_df = pd.DataFrame(age, columns=['age'])
            
            bins = np.linspace(0, 150, 10).tolist() + [np.inf]
            selection_df.loc[:, 'age_bin'] = pd.cut(selection_df['age'], bins=bins)

            indexes, _ = train_test_split(
                selection_df,
                train_size=self.fraction,
                stratify=selection_df['age_bin'],
                random_state=42 # same random seed as overall 
            ) 
            indexes = indexes.index.to_list()

            self.indexes = indexes 

    def __len__(self):
        if self.split == 'train' and self.fraction < 1.0:
            return len(self.indexes)
        return self.len 

    def __getitem__(self, idx): 
        if self.split == 'train' and self.fraction < 1.0:
            idx = self.indexes[idx]

        with h5py.File(self.data_file, 'r', swmr=True) as f: 
            cloud = f[self.split]['clouds'][idx]
            age = f[self.split]['age'][idx]

        cloud = normalize_pointcloud(cloud)
        cloud = translate_pointcloud(cloud) 
        cloud = jitter_pointcloud(cloud)
        cloud = rotate_pointcloud(cloud)
        cloud = flip_pointcloud(cloud)


        return cloud, age
    

class ALS_50K(Dataset): 
    def __init__(self, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.data_file = Path('/share/projects/erasmus/hansend/thesis/data/pretraining/ALS_50K.h5')
        
        with h5py.File(self.data_file, 'r', swmr=True) as f:
            self.len = f['data']['instance_xyz'].shape[0]
                
    def __len__(self):
        return self.len 

    def __getitem__(self, idx): 
        with h5py.File(self.data_file, 'r', swmr=True) as f: 
            instance_xyz = f['data']['instance_xyz'][idx] 
            instance_xyz = instance_xyz.reshape(-1, 3) # (N, 3) 
            
            # FPS supsampling to num_points
            if instance_xyz.shape[0] > self.num_points:
                instance_idxs = fpsample.bucket_fps_kdline_sampling(instance_xyz, self.num_points, h=3)
                instance_xyz = instance_xyz[instance_idxs]
                #instance_nheights = instance_nheights[instance_idxs]

                # augmentations with jittering as it is not used here before 
                # instance_xyz = rotate_pointcloud(instance_xyz)
                # instance_xyz = translate_pointcloud(instance_xyz)
                # instance_xyz = jitter_pointcloud(instance_xyz)
                # instance_xyz = flip_pointcloud(instance_xyz)
                instance_xyz = normalize_pointcloud(instance_xyz)


            # adding jittered points to num_points
            elif instance_xyz.shape[0] < self.num_points: 
                point_diff = self.num_points - instance_xyz.shape[0]
                idxs = np.random.choice(instance_xyz.shape[0], point_diff, replace=True)
                add_points_xyz = instance_xyz[idxs]
                instance_xyz = np.concatenate((instance_xyz, add_points_xyz), axis=0)
            
                # augmentations without jitter as it has beend done before to upsample to num_points 
                # instance_xyz = rotate_pointcloud(instance_xyz)
                # instance_xyz = translate_pointcloud(instance_xyz)
                # instance_xyz = flip_pointcloud(instance_xyz)
                instance_xyz = normalize_pointcloud(instance_xyz)
        

        return instance_xyz, idx #, instance_nheights
