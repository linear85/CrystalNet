#!/bin/sh -l
# FILENAME:  Python example

#SBATCH -A mat210034-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name gnnMC_model

module load anaconda
conda activate pytorch

python graphconv.py

