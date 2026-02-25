#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=logs/optuna.out
#SBATCH --error=logs/optuna.err
#SBATCH --time=6-12:00:00
#SBATCH --partition=gpu        # or 'cpu' depending on your cluster
#SBATCH --gres=gpu:1           # request 1 GPU, remove if CPU only

module load apptainer/1.3.5

# Activate your environment
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache

# nvidia-smi

apptainer exec \
    --nv \
    --bind /data:/data \
    --bind $PWD:/my-work-dir \
    vgg-faiss-gpu.sif \
    python -u /my-work-dir/main.py --task "ssl_hyptun" --config "config/ssl_hyptun.yaml"