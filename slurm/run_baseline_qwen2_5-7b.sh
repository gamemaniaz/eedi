#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --job-name=run_baseline-qwen2_5-7b
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
CONDA_ENV=eedi2
$HOME/miniforge3/envs/$CONDA_ENV/bin/python baseline-qwen2_5-7b.py
