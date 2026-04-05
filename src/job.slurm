#!/bin/bash
#SBATCH --job-name=main         # Job name
#SBATCH --output=logs/job%j.out                # Standard output file
#SBATCH --error=logs/job%j.err                 # Standard error file
#SBATCH --time=2-00:00:00                 # Max runtime (hh:mm:ss)
#SBATCH --partition=std                 # Partition/queue name

# Load modules if needed (depends on your cluster setup)
module load conda

# Activate your virtual environment
conda activate python3.14

# Run your Python script
python -u main.py --task "temp"
    # ["enhance_images", "labeling", "preprocess_incoming", "show_zip_contents", "temp"]