#!/bin/bash
#
#SBATCH --partition=gpu_min11gb    # Debug partition
#SBATCH --qos=gpu_min11gb_ext          # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_baseline     # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)

CUDA_LAUNCH_BLOCKING=1 python3 dinov2_large.py \
    --num_classes 2 \
    --epochs 50 \
    --classification 'insurance' \
    --augmentation True 


