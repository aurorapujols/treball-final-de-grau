#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err
#SBATCH --time=1-12:00:00
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
    python -u training/hyperparameter_tunning/visualizations.py