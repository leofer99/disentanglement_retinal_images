#!/bin/bash
#
#SBATCH --partition=gpu_min11gb    # Debug partition
#SBATCH --qos=gpu_min11gb_ext          # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_aug    # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)


#Non-aug Conv Age
CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
    --latent_dim 256 \
    --batch_size 32 \
    --num_classes 2 \
    --folder_name 'bal_augFalse_convnext_id_age_DisenTrue_ld256_bs32_lrG5e-05_DFall' \
    --model_epoch 80 \
    --model 'convnext'

#Non-aug DINO Sex

#Non-aug DINO Insurance

#Non-aug ConvNeXt Sex128

#Non-aug ConvNeXt Sex64

#Non-aug ConvNeXt Sex32







# ## Disentanglement True:
# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 150 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'age' \
#     --model 'convnext' \
#     --disentanglement True \
#     --augmentation False


# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 150 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'dinov' \
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
#     --model 'dinov' \
#     --disentanglement False \
#     --augmentation True



# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 128 \
#     --batch_size 32 \
#     --epochs 120 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'convnext' \
#     --disentanglement True \
#     --augmentation True

#     CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 64 \
#     --batch_size 32 \
#     --epochs 120 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'convnext' \
#     --disentanglement True \
#     --augmentation True


#     CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 32 \
#     --batch_size 32 \
#     --epochs 120 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'convnext' \
#     --disentanglement True \
#     --augmentation True



# CUDA_LAUNCH_BLOCKING=1 python3 disentanglement_network_mBRSET.py \
#     --num_classes 2 \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --epochs 200 \
#     --df_size 'all' \
#     --learning_rate_gen 5e-5 \
#     --id_info 'sex' \
#     --model 'convnext' \
#     --disentanglement True \
#     --augmentation True



##disentanglement False:

