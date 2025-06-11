#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --partition shortrun
#SBATCH --output train_enriched_mask_nm.txt
#SBATCH --time=0-04:00:00
#SBATCH --j nm_train 


setcuda 12.1
conda activate mae3d

python 