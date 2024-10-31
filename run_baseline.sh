#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --job-name=run_baseline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
$HOME/miniforge3/condabin/mamba env update -f environment.yml -y
$HOME/miniforge3/envs/eedi2/bin/python baseline2.py