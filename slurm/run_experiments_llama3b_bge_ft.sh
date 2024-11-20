#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --job-name=run_baseline-llama3_1-8b
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

python eedi/run.py -m llama3b -k none -e bge-ft
python eedi/run.py -m llama3b -k genk -e bge-ft
python eedi/run.py -m llama3b -k tot -e bge-ft
python eedi/run.py -m llama3b -k rag -e bge-ft
