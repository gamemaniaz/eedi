#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --job-name=run_install_env
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
CONDA_ENV=eedi2
$HOME/miniforge3/condabin/mamba env create -n $CONDA_ENV -f environment.yml -y
