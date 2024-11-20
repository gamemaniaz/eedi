#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --job-name=run_install_env
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
mamba env create -n eedi -f environment.yml -y
