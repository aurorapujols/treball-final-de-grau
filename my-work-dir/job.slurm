#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=logs/temp.out
#SBATCH --error=logs/temp.err
#SBATCH --time=1-12:00:00
#SBATCH --partition=std        # or 'cpu' depending on your cluster
# #SBATCH --gres=gpu:1           # request 1 GPU, remove if CPU only

module load apptainer/1.3.5

# Activate your environment
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache

# nvidia-smi

apptainer exec \
    --nv \
    --bind /data:/data \
    --bind $PWD:/my-work-dir \
    docker-env.sif \
    python -u /my-work-dir/main.py --task "temp" --config "config/temp.yaml"
    # python -u training/hyperparameter_tunning/visualizations.py