#!/bin/sh
#SBATCH --time=5-00:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --job-name=run_baseline-llama3_1-8b
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1374073@u.nus.edu
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

python eedi/run.py -m qwen7b -k none -e bge
python eedi/run.py -m qwen7b -k genk -e bge
python eedi/run.py -m qwen7b -k tot -e bge
python eedi/run.py -m qwen7b -k rag -e bge