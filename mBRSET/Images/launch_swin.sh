#!/bin/bash
#
#SBATCH --partition=gpu_min24gb    # Debug partition gpu_min11gb
#SBATCH --qos=gpu_min24gb_ext          # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_swin     # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)
CUDA_LAUNCH_BLOCKING=1 python3 swinv2.py \
    --num_classes 2 \
    --epochs 50 \
    --classification '2class_DR'  #'2class_DR', '3class_DR', 'gender'


# CUDA_LAUNCH_BLOCKING=1 python3 swinv2.py \
#     --num_classes 2 \
#     --epochs 50 \
#     --classification 'gender'  #'2class_DR', '3class_DR', 'gender'

    
# CUDA_LAUNCH_BLOCKING=1 python3 swinv2.py \
#     --num_classes 3 \
#     --epochs 50 \
#     --classification '3class_DR'  #'2class_DR', '3class_DR', 'gender'






