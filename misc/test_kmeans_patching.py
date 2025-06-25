import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pointnet2_ops.pointnet2_utils import furthest_point_sample
import sys 
sys.path.append('../')

from data_yours import ALS_50K

from torch_kmeans import KMeans, SoftKMeans, ConstrainedKMeans


def gather_patches(points, patch_labels, selected_patch_ids, nsample):
    """
    points: [B, N, C] — original point cloud
    patch_labels: [B, N] — cluster label for each point
    selected_patch_ids: [B, P] — selected patch indices
    nsample: int — number of points per patch

    Returns:
        gathered_patches: [B, P, nsample, C]
    """
    B, N, C = points.shape
    P = selected_patch_ids.shape[1]
    gathered_patches = []

    for b in range(B):
        point_set = points[b]       # [N, C]
        labels = patch_labels[b]    # [N]
        selected_ids = selected_patch_ids[b]  # [P]
        patches = []

        for pid in selected_ids:
            idx = (labels == pid).nonzero(as_tuple=True)[0]
            n = len(idx)
    
            if n < nsample:
                # Pad by repeating random indices
                pad_idx = idx[torch.randint(0, n, (nsample - n,), device=points.device)]
                idx = torch.cat([idx, pad_idx])
            elif n > nsample:
                sub_idx = furthest_point_sample(point_set[idx].unsqueeze(0), nsample)  # [1, nsample]
                idx = sub_idx.squeeze(0)
            # else: n == nsample, do nothing

            patch = point_set[idx]  # [nsample, C]
            patches.append(patch)

        patches = torch.stack(patches, dim=0)  # [P, nsample, C]
        gathered_patches.append(patches)

    return torch.stack(gathered_patches, dim=0)  # [B, P, nsample, C]


def split_kmeans_patches(batch, mask_ratio=0.7, nsample=32):
    B, N, C = batch.shape 

    # calculate number of patches
    num_patches = N // nsample # assuming N is num_points
    num_masked_patches = int(num_patches * mask_ratio) 
    num_vis_patches = num_patches - num_masked_patches

    # generate clusters 
    kmeans_model = KMeans(n_clusters=num_patches) 
    cluster_results = kmeans_model(batch)

    cl_idx = cluster_results.labels 
    cl_centroids = cluster_results.centers 

    shuffle_idx = torch.randperm(num_patches*B).reshape(B, num_patches).argsort(dim=1)
    vis_patch_idx, mask_patch_idx = shuffle_idx[:, :num_vis_patches], shuffle_idx[:, num_vis_patches:] 

    vis_center_pos = torch.gather(cl_centroids, 1, vis_patch_idx.unsqueeze(-1).expand(-1, -1, cl_centroids.shape[-1]).cuda())
    mask_center_pos = torch.gather(cl_centroids, 1, mask_patch_idx.unsqueeze(-1).expand(-1, -1, cl_centroids.shape[-1]).cuda())

    # gather pos 
    vis_pos = gather_patches(batch, cl_idx, vis_patch_idx, nsample)
    mask_pos = gather_patches(batch, cl_idx, mask_patch_idx, nsample)


    return mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx


def main():
    ds = ALS_50K(num_points=2048)

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    batch = next(iter(dl))

    batch = batch[0]

    batch = batch.cuda()

    mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx = split_kmeans_patches(batch)

    print(
        f'Mask patches: {mask_pos.shape}, Vis patches: {vis_pos.shape},\n' 
        f'Mask centers: {mask_center_pos.shape}, Vis centers: {vis_center_pos.shape},\n' 
        f'Mask patch idx: {mask_patch_idx.shape}, Vis patch idx: {vis_patch_idx.shape},\n'
        f'Shuffle idx: {shuffle_idx.shape}'
    )


if __name__ == '__main__':
    main()

