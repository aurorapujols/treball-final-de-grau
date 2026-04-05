#!/bin/bash
#SBATCH --job-name=oct-dec2023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-6:00:00
#SBATCH --output=logs/oct-dec2023_avi.log 
#SBATCH --error=logs/oct-dec2023_avi.err

# Load environment
module load conda
conda activate python3.14

7z a -t7z -mmt=$SLURM_CPUS_PER_TASK -mx=9 /data/upftfg26/apujols/raw_data/videos/oct-dec2023.7z /data/upftfg26/apujols/raw_data/videos/oct-dec2023