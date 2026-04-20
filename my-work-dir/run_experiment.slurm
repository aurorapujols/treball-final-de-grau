#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=logs/plots/plots1.1.out
#SBATCH --error=logs/plots/plots1.1.err
#SBATCH --time=1-00:00:00
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
    docker-env.sif \
    python -u /my-work-dir/main.py --task "plot_results" --config "config/plotting.yaml"

    # python -u /my-work-dir/main.py --task "train_classifiers" --config "config/classifiers.yaml"

    # python -u /my-work-dir/main.py --task "train_classifiers" --config "config/classifiers.yaml"
    # python -u /my-work-dir/main.py --task "ssl_final_model" --config "config/ssl_finalmodel.yaml"
    # python -u /my-work-dir/main.py --task "ssl_hyptun" --config "config/ssl_hyptun.yaml"
    # python -u /my-work-dir/main.py --task "ssl_architecture" --config "config/ssl_architecture.yaml"