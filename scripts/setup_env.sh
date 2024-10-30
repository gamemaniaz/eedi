#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 5-00:00:00
#SBATCH -c 20
#SBATCH --gpus=h100-47:1
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tienkhoa@comp.nus.edu.sg

cd $HOME/coursework/eedi
conda env create -f environment.yml -y
# pip install -r requirements.txt
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

