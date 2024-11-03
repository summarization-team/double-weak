#!/bin/bash

source /mmfs1/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/clmbr/babylm24/envs/gpu-double-weak

# error message details
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Disable Weights and Biases login requirements
export WANDB_DISABLED="true"

# Set Transformers Caches to clmbr gscratch to avoid issues with disk space in user accounts
export HF_HOME=../.cache/transformers
export HF_DATASETS_CACHE=../.cache/datasets


# Build shell command
python src/analyze.py