#!/bin/bash
#
#SBATCH --partition=gpu_min11gb    # Debug partition
#SBATCH --qos=gpu_min11gb_ext          # Debug QoS level, add _ext!
#SBATCH --job-name=mBRSET_aug    # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output


# Commands / scripts to run (e.g., python3 train.py)
# (...)


# ## Disentanglement True:
CUDA_LAUNCH_BLOCKING=1 python3 resume_training_mBRSET.py 




