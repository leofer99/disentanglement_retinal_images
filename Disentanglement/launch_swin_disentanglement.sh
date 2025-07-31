#!/bin/bash
#
#SBATCH --partition=gpu_min80gb    # Debug partition
#SBATCH --qos=gpu_min80gb          # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_aug    # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)


# Swin:
CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
    --num_classes 2 \
    --latent_dim 256 \
    --batch_size 32 \
    --epochs 150 \
    --df_size 'all' \
    --learning_rate_gen 5e-5 \
    --id_info 'sex' \
    --model 'swin' \
    --disentanglement False \
    --augmentation True

CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
    --num_classes 2 \
    --latent_dim 256 \
    --batch_size 32 \
    --epochs 150 \
    --df_size 'all' \
    --learning_rate_gen 5e-5 \
    --id_info 'educational_level' \
    --model 'swin' \
    --disentanglement False \
    --augmentation True

CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
    --num_classes 2 \
    --latent_dim 256 \
    --batch_size 32 \
    --epochs 150 \
    --df_size 'all' \
    --learning_rate_gen 5e-5 \
    --id_info 'insurance' \
    --model 'swin' \
    --disentanglement False \
    --augmentation True





# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 150 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'swin' \
#     --disentanglement False \
#     --augmentation True


# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 150 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'educational_level' \
#     --model 'swin' \
#     --disentanglement False \
#     --augmentation True


# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 150 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'insurance' \
#     --model 'swin' \
#     --disentanglement False \
#     --augmentation True
