import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import numpy as np

from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, get_laplacian

from torch_kmeans import KMeans

from pointnet2_ops.pointnet2_utils import furthest_point_sample



class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts, mode='sum'):
        P = self.batch_pairwise_dist(gts, preds)
        mins_1, _ = torch.min(P, 1)
        mins_2, _ = torch.min(P, 2)
        if mode == 'mean':
            loss_1 = torch.mean(mins_1)
            loss_2 = torch.mean(mins_2)
        elif mode == 'sum':
            loss_1 = torch.sum(mins_1)
            loss_2 = torch.sum(mins_2)
        # print(mins.shape)
        # exit(0)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.long()
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss    


def cal_loss_cd(pred, label, mode='sum'):
    pred = pred.permute(0, 2, 1)
    label = label.permute(0, 2, 1)
    chamfer_dist = ChamferLoss()
    loss = chamfer_dist(pred, label, mode)

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def write_plyfile(file_name, point_cloud):
    f = open(file_name + '.ply', 'w')
    init_str = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(point_cloud)) + \
               "\nproperty float x\nproperty float y\nproperty float z\n" \
               "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(init_str)
    for i in range(len(point_cloud)):
        f.write(str(round(float(point_cloud[i][0]), 6)) + ' ' + str(round(float(point_cloud[i][1]), 6)) + ' ' +
                str(round(float(point_cloud[i][2]), 6)) + '\n')
    f.close()


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


def split_knn_patches(xyz, mask_ratio=0.7, nsample=32, random=True):
    B, N, C = xyz.shape
    npoint = N // nsample
    device = xyz.device

    num_patches = npoint
    num_mask = int(mask_ratio * num_patches)
    num_vis = num_patches - num_mask

    center_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()
    center_pos = index_points(xyz, center_idx)

    if random:
        shuffle_idx = torch.rand(num_patches, device=device).argsort()
        vis_patch_idx, mask_patch_idx = shuffle_idx[:num_vis], shuffle_idx[num_vis:]

        mask_center_idx = center_idx[:, mask_patch_idx]
        vis_center_idx = center_idx[:, vis_patch_idx]
        mask_center_pos = index_points(xyz, mask_center_idx)
        vis_center_pos = index_points(xyz, vis_center_idx)

        all_patch_idx = knn_point(nsample, xyz, center_pos)  # [B, num_patches, nsample]
        mask_idx = all_patch_idx[:, mask_patch_idx]
        vis_idx = all_patch_idx[:, vis_patch_idx]
        mask_pos = index_points(xyz, mask_idx)  # [B, num_mask, nsample, C]
        vis_pos = index_points(xyz, vis_idx)  # [B, num_vis, nsample, C]
    else:
        mask_point_idx = np.random.randint(num_patches)
        mask_point_pos = center_pos[:, mask_point_idx]
        mask_patch_idx = knn_point(num_mask, center_pos, mask_point_pos.unsqueeze(1)).squeeze()
        vis_patch_idx = torch.empty((B, num_vis), device=device, dtype=int)
        for b in range(B):
            idx_all = set(np.arange(num_patches, dtype=int))
            mask_idx = set(mask_patch_idx[b].tolist())
            vis_idx = idx_all - mask_idx
            vis_patch_idx[b] = torch.tensor(list(vis_idx), device=device, dtype=torch.long)

        shuffle_idx = torch.cat((vis_patch_idx, mask_patch_idx), dim=1).to(device)

        batch_idx = torch.arange(B, device=device).unsqueeze(-1)
        mask_center_idx = center_idx[batch_idx, mask_patch_idx]
        vis_center_idx = center_idx[batch_idx, vis_patch_idx]
        mask_center_pos = index_points(xyz, mask_center_idx)
        vis_center_pos = index_points(xyz, vis_center_idx)

        all_patch_idx = knn_point(nsample, xyz, center_pos)  # [B, num_patches, nsample]
        mask_idx = all_patch_idx[batch_idx, mask_patch_idx]
        vis_idx = all_patch_idx[batch_idx, vis_patch_idx]
        mask_pos = index_points(xyz, mask_idx)  # [B, num_mask, nsample, C]
        vis_pos = index_points(xyz, vis_idx)  # [B, num_vis, nsample, C]

    return mask_pos,  vis_pos,  mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx

def get_kNN_patch_idxs(xyz, center_xyz, k):
    dists = torch.cdist(xyz, center_xyz)

    _, knn_idxs = torch.topk(dists, k, largest=False, dim=0) # shape: (N points, 3)

    return knn_idxs


def center_split_masking(batch, masking_ratio=0.2, patch_size=16): 
    B, N, C = batch.shape 

    # calculate number of patches
    num_patches = N // patch_size # assuming N is num_points
    num_masked_patches = int(num_patches * masking_ratio) 
    num_vis_patches = num_patches - num_masked_patches

    vis_pos = []
    mask_pos = [] 
    mask_center_pos = [] 
    vis_center_pos = [] 
    # generate patch_idx and shuffle_idx later 

    for i in range(B): 
        pc = batch[i]

        # generate random T 
        if np.random.choice([True, False]): 
            t, axis = torch.median(pc[:,0]), 'x'
        else: 
            t, axis = torch.median(pc[:, 1]), 'y'
        
        # subset the point cloud
        if axis == 'x':
            pc_vis = pc[pc[:, 0] > t] 
            pc_masked = pc[pc[:, 0] <= t]
        elif axis == 'y': 
            pc_vis = pc[pc[:, 1] <= t] 
            pc_masked = pc[pc[:, 1] > t]

        # find patch centers in the subsets 
        vis_centers = pointnet2_utils.furthest_point_sample(pc_vis.cuda().unsqueeze(0), num_vis_patches).cpu().squeeze(0)
        masked_centers = pointnet2_utils.furthest_point_sample(pc_masked.cuda().unsqueeze(0), num_masked_patches).cpu().squeeze(0)

        # select center points 
        vis_center_points = pc_vis[vis_centers]
        masked_center_points = pc_masked[masked_centers]

        # get patch idxs
        vis_patch_idxs = get_kNN_patch_idxs(pc_vis, vis_center_points, k=patch_size)
        masked_patch_idxs = get_kNN_patch_idxs(pc_masked, masked_center_points, k=patch_size)

        # get patches 
        vis_patches = [pc_vis[vis_patch_idxs[:, i]] for i in range(vis_patch_idxs.shape[1])]
        masked_patches = [pc_masked[masked_patch_idxs[:, i]] for i in range(masked_patch_idxs.shape[1])]

        vis_patches_tensor = torch.stack(vis_patches, dim=0)
        masked_patches_tensor = torch.stack(masked_patches, dim=0)

        vis_pos.append(vis_patches_tensor)
        vis_center_pos.append(vis_center_points)
        mask_pos.append(masked_patches_tensor)
        mask_center_pos.append(masked_center_points)

    vis_pos = torch.stack(vis_pos, dim=0)
    vis_center_pos = torch.stack(vis_center_pos, dim=0)
    mask_pos = torch.stack(mask_pos, dim=0)
    mask_center_pos = torch.stack(mask_center_pos, dim=0)

    # changed here 
    idx_all = torch.arange(0, num_patches)
    vis_patch_idx = idx_all[:num_vis_patches]
    mask_patch_idx = idx_all[num_vis_patches:]

    shuffle_idx = torch.cat((vis_patch_idx, mask_patch_idx), dim=0)

    return mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx


def quadrant_masking(batch, masking_ratio=0.2, patch_size=16): 
    B, N, C = batch.shape 

    num_patches = N // patch_size
    num_masked_patches = int(num_patches * masking_ratio)
    num_vis_patches = num_patches - num_masked_patches

    vis_pos, mask_pos = [], []
    vis_center_pos, mask_center_pos = [], []

    for i in range(B): 
        pc = batch[i]
        t_x = torch.median(pc[:,0])
        t_y = torch.median(pc[:, 1])

        # Split into quadrants
        quads = [
            pc[(pc[:, 0] > t_x) & (pc[:, 1] >= t_y)],  # upper right
            pc[(pc[:, 0] <= t_x) & (pc[:, 1] >= t_y)], # upper left
            pc[(pc[:, 0] > t_x) & (pc[:, 1] < t_y)],   # lower right
            pc[(pc[:, 0] <= t_x) & (pc[:, 1] < t_y)]   # lower left
        ]

        # Always select exactly 2 quadrants as masked (1) and 2 as visible (0)
        select_idx = np.zeros(4, dtype=int)
        select_idx[:2] = 1
        np.random.shuffle(select_idx)
        masked_quads = [q for q, s in zip(quads, select_idx) if s == 1] # make sure there is at least 1 patch
        visible_quads = [q for q, s in zip(quads, select_idx) if s == 0]

        #  check if there are points in each quadrant
        masked_quads = [q for q in masked_quads if q.shape[0] > 0]
        visible_quads = [q for q in visible_quads if q.shape[0] > 0]
                        

        # fall back to the largest quadrant if not enough points in one quadrant
        if len(masked_quads) == 1: # or sum([q.shape[0] for q in masked_quads]) < patch_size * (num_masked_patches):
            masked_quads = [torch.cat(masked_quads)]

        if len(visible_quads) == 1: # or sum([q.shape[0] for q in visible_quads]) < patch_size * (num_vis_patches):
            visible_quads = [torch.cat(visible_quads)]


        # # Upsample quadrants if needed to ensure enough points for patching
        for idx, q in enumerate(masked_quads): 
            to_add = patch_size - q.shape[0]
            if q.shape[0] > 0 and to_add > 0:
                dupl_idx = torch.randint(0, len(q), (to_add,), device=q.device)
                masked_quads[idx] = torch.cat((q, q[dupl_idx]), dim=0)
        for idx, q in enumerate(visible_quads):
            to_add = patch_size - q.shape[0]
            if q.shape[0] > 0 and to_add > 0:
                dupl_idx = torch.randint(0, len(q), (to_add,), device=q.device)
                visible_quads[idx] = torch.cat((q, q[dupl_idx]), dim=0)

        # Distribute patches as evenly as possibel
        patches_per_masked = [num_masked_patches // len(masked_quads) + (1 if x < num_masked_patches % 2 else 0) for x in range(2)]
        patches_per_visible = [num_vis_patches // len(visible_quads) + (1 if x < num_vis_patches % 2 else 0) for x in range(2)] 
        
        # get center points and patches masked
        masked_centers = [] 
        masked_patches = []
        for q, n_p in zip(masked_quads, patches_per_masked):
            # center points
            centers = pointnet2_utils.furthest_point_sample(q.unsqueeze(0), n_p).squeeze(0)
            center_points = q[centers]
            masked_centers.append(center_points)
            # patches
            patches_idx = get_kNN_patch_idxs(q, center_points, k=patch_size)
            patches = [q[patches_idx[:, j]] for j in range(patches_idx.shape[1])]
            masked_patches.extend(patches)  
        # get center points and patches vis 
        vis_centers = [] 
        vis_patches = [] 
        for q, n_p in zip(visible_quads, patches_per_visible):
            # center points
            centers = pointnet2_utils.furthest_point_sample(q.unsqueeze(0), n_p).squeeze(0)
            center_points = q[centers]
            vis_centers.append(center_points)
            # patches
            patches_idx = get_kNN_patch_idxs(q, center_points, k=patch_size)
            patches = [q[patches_idx[:, j]] for j in range(patches_idx.shape[1])]
            vis_patches.extend(patches)
        vis_pos.append(torch.stack(vis_patches, dim=0))
        mask_pos.append(torch.stack(masked_patches, dim=0))
        vis_center_pos.append(torch.cat(vis_centers, dim=0))
        mask_center_pos.append(torch.cat(masked_centers, dim=0))
    
    vis_pos = torch.stack(vis_pos, dim=0)
    mask_pos = torch.stack(mask_pos, dim=0)
    vis_center_pos = torch.stack(vis_center_pos, dim=0)
    mask_center_pos = torch.stack(mask_center_pos, dim=0)

    idx_all = torch.rand(num_patches).argsort()
    vis_patch_idx = idx_all[:num_vis_patches]
    mask_patch_idx = idx_all[num_vis_patches:]
    shuffle_idx = torch.cat((vis_patch_idx, mask_patch_idx), dim=0)

    return mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx


def filter_signal_1d(f, eigenvectors, cutoff_ratio=0.8):
    """
    f: 1D torch tensor of shape (num_points,) representing a single coordinate (x, y, or z)
    eigenvectors: the matrix of eigenvectors from Laplacian decomposition
    cutoff_ratio: fraction of low frequencies to keep (e.g. 0.8 => keep 80%, remove top 20%)
    returns: f_filtered, the filtered version of the signal
    """
    # Graph Fourier Transform
    if f.is_cuda or eigenvectors.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    f = f.to(device)
    eigenvectors = eigenvectors.to(device)

    f_hat = torch.matmul(eigenvectors.T, f)
    num_components = f_hat.shape[0]
    cutoff = int(cutoff_ratio * num_components)

    # Zero out the highest frequencies
    f_hat_filtered = f_hat.clone()
    f_hat_filtered[cutoff:] = 0

    # Inverse GFT
    f_filtered = torch.matmul(eigenvectors, f_hat_filtered)
    return f_filtered

def frequency_patching(pointcloud, k=10, cutoff_ratio=0.8, patch_size=16, mask_ratio=0.5): 

    B, N, C = pointcloud.shape

    num_patches = N // patch_size 
    num_masked_patches = int(num_patches * mask_ratio) 
    num_vis_patches = num_patches - num_masked_patches

    vis_patches_batch = [] 
    mask_patches_batch = []
    vis_center_pos_batch = []
    mask_center_pos_batch = []

    for b in range(B):

        points = pointcloud[b]  # shape: (N, C)
        points_np = points.cpu().numpy()  # Convert to numpy for sklearn
        adj_matrix = kneighbors_graph(points_np, k, mode='connectivity', include_self=False)

        edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
        num_nodes = points.shape[0]

        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()

        L = (L + L.T) / 2
        L += 1e-6 * torch.eye(num_nodes)

        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Extract each coordinate as its own signal
        f_x = points[:, 0]
        f_y = points[:, 1]
        f_z = points[:, 2]


        # Define cutoff ratio (keep lower 80%, remove top 20% frequencies)
        cutoff_ratio = cutoff_ratio

        # Filter each coordinate
        f_x_filtered = filter_signal_1d(f_x, eigenvectors, cutoff_ratio) # needs tensor
        f_y_filtered = filter_signal_1d(f_y, eigenvectors, cutoff_ratio)
        f_z_filtered = filter_signal_1d(f_z, eigenvectors, cutoff_ratio)

        # Construct the filtered point cloud
        filtered_points = points.clone()
        filtered_points[:, 0] = f_x_filtered
        filtered_points[:, 1] = f_y_filtered
        filtered_points[:, 2] = f_z_filtered
        
        # calculate 3D difference between original and filtered points
        difference = torch.abs(points - filtered_points)

        t = torch.quantile(difference, 0.9)

        mask = difference > t

        masked_points = points[mask.any(dim=1)]
        vis_points = points[~mask.any(dim=1)]


        # get patches based on setting 
        vis_center_idx = pointnet2_utils.furthest_point_sample(vis_points.unsqueeze(0), num_vis_patches).squeeze(0)
        vis_center_pos = vis_points[vis_center_idx] 

        mask_center_idx = pointnet2_utils.furthest_point_sample(vis_points.unsqueeze(0), num_vis_patches).squeeze(0)
        mask_center_pos = masked_points[mask_center_idx]

        # get patches based on kNN 
        patch_idx = knn_point(patch_size, vis_points.unsqueeze(0), vis_center_pos.unsqueeze(0)).squeeze(0)
        vis_patches = index_points(vis_points.unsqueeze(0), patch_idx.unsqueeze(0)).squeeze(0)

        patch_idx = knn_point(patch_size, masked_points.unsqueeze(0), mask_center_pos.unsqueeze(0)).squeeze(0)
        mask_patches = index_points(masked_points.unsqueeze(0), patch_idx.unsqueeze(0)).squeeze(0)

        vis_patches_batch.append(vis_patches)
        mask_patches_batch.append(mask_patches)
        vis_center_pos_batch.append(vis_center_pos)
        mask_center_pos_batch.append(mask_center_pos)

    # Stack the results for the batch
    vis_pos = torch.stack(vis_patches_batch, dim=0)  
    mask_pos = torch.stack(mask_patches_batch, dim=0)
    vis_center_pos = torch.stack(vis_center_pos_batch, dim=0)
    mask_center_pos= torch.stack(mask_center_pos_batch, dim=0)

    idx_all = torch.arange(0,num_patches)
    vis_patch_idx = idx_all[:num_vis_patches]
    mask_patch_idx = idx_all[num_vis_patches:]
    shuffle_idx = torch.cat((vis_patch_idx, mask_patch_idx), dim=0)

    return mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx





def gather_patches(points, patch_labels, selected_patch_ids, nsample, expected_patches):
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
            if n == 0: 
                continue
    
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

        if patches.shape[0] < expected_patches: 
            pad_idx = torch.randint(0, patches.shape[0], (expected_patches - patches.shape[0],), device=points.device)
            pad_patches = patches[pad_idx]  # [pad, nsample, C]
            patches = torch.cat([patches, pad_patches], dim=0)

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
    vis_pos = gather_patches(batch, cl_idx, vis_patch_idx, nsample, num_vis_patches)
    mask_pos = gather_patches(batch, cl_idx, mask_patch_idx, nsample, num_masked_patches)


    return mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx


class stepwise_scheduler():
    def __init__(self, schedule, default_value=None): 
        self.schedule = schedule 
        self.default_value = default_value

        self.schedule = sorted(schedule, key=lambda x: x[0])

        if self.default_value is None: 
            self.default_value = self.schedule[-1][1] # first value as default 

    def __call__(self, epoch): 
        for i, (e, v) in enumerate(self.schedule): 
            if epoch < e:
                return self.schedule[i - 1][1] 
            elif (epoch > e or epoch == e) and i == len(self.schedule) - 1:
                return v