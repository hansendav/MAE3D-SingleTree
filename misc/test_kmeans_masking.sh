#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --partition shortrun
#SBATCH --output test_kmeans_patching
#SBATCH --time=0-00:10:00
#SBATCH --j ft_als50k



setcuda 12.1
conda activate mae3d

cd /share/home/e2405193/MAE3D-SingleTree/misc

python test_kmeans_patching.py 