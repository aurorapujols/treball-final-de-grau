#!/bin/bash
#SBATCH --job-name=february2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-6:00:00
#SBATCH --output=logs/february2026_avi.log 
#SBATCH --error=logs/february2026_avi.err

# Load environment
module load conda
conda activate python3.14

7z a -t7z -mmt=$SLURM_CPUS_PER_TASK -mx=9 /data/upftfg26/apujols/raw_data/videos/february2026.7z /data/upftfg26/apujols/raw_data/videos/february2026