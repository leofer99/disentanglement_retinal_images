#!/bin/bash
#
#SBATCH --partition=gpu_min8gb    # Debug partition gpu_min11gb
#SBATCH --qos=gpu_min8gb_ext          # Debug QoS level, add _ext! gpu_min11gb_ext
#SBATCH --job-name=mBRSET_fmodel     # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)
CUDA_LAUNCH_BLOCKING=1 python3 train_feature_models_mbrset_less.py \




