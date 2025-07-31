#!/bin/bash

#SBATCH --partition=gpu_min11gb      # Debug partition
#SBATCH --qos=gpu_min11gb_ext           # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_test    # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)

# simplified version
# CUDA_LAUNCH_BLOCKING=1 python3 baseline_test_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'baseline_augTrue_convnext_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall' \
#     --model_epoch 100 \
#     --model 'convnext'



#mBRSET
CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
    --latent_dim 256 \
    --batch_size 32 \
    --num_classes 2 \
    --folder_name 'dinov_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
    --model_epoch 100 \
    --model 'dinov'





# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_BRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_dinov_id_patient_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall' \
#     --model_epoch 80 \
#     --model 'dinov'


# CUDA_LAUNCH_BLOCKING=1 python3 re_plot__loss_graphs.py \


# ##AUGMENTED
# #Convnext:
# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 128 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_convnext_id_sex_DisenTrue_ld128_bs32_lrG5e-05_DFall' \
#     --model_epoch 120 \
#     --model 'convnext'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 64 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_convnext_id_sex_DisenTrue_ld64_bs32_lrG5e-05_DFall' \
#     --model_epoch 120 \
#     --model 'convnext'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 32 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_convnext_id_sex_DisenTrue_ld32_bs32_lrG5e-05_DFall' \
#     --model_epoch 120 \
#     --model 'convnext'




# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_dinov_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall' \
#     --model_epoch 200 \
#     --model 'dinov'


# #Swin:
# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 16 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_swin_id_sex_DisenTrue_ld256_bs16_lrG5e-05_DFall' \
#     --model_epoch 150 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 16 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_swin_id_insurance_DisenTrue_ld256_bs16_lrG5e-05_DFall' \
#     --model_epoch 150 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 16 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_swin_id_educational_level_DisenTrue_ld256_bs16_lrG5e-05_DFall' \
#     --model_epoch 150 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'bal_augTrue_swin_id_sex_DisenFalse_ld256_bs16_lrG5e-05_DFall' \
#     --model_epoch 150 \
#     --model 'swin'


# # NOT AUGMENTED
# #Convnext:
# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'convnext_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 80 \
#     --model 'convnext'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'convnext_id_sex_DisenFalse_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 35 \
#     --model 'convnext'

# #DINO:
# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 128 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'dinov_id_insurance_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 80 \
#     --model 'dinov'


# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 32 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'dinov_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 100 \
#     --model 'dinov'



# #Swin:
# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_educational_level_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 100 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_educational_level_DisenFalse_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 120 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_insurance_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 120 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_insurance_DisenFalse_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 120 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_sex_DisenTrue_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 120 \
#     --model 'swin'

# CUDA_LAUNCH_BLOCKING=1 python3 test_disentangled_mBRSET.py \
#     --latent_dim 256 \
#     --batch_size 32 \
#     --num_classes 2 \
#     --folder_name 'swin_id_sex_DisenFalse_ld256_bs32_lrG5e-05_DFall_augmentationFalse' \
#     --model_epoch 120 \
#     --model 'swin'
