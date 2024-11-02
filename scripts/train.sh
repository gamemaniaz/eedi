#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 5-00:00:00
#SBATCH -c 20
#SBATCH --gpus=h100-96:1
#SBATCH --mem=160G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tienkhoa@comp.nus.edu.sg

TMPDIR=`mktemp -d`
WORK_DIR=$HOME/coursework/eedi

cd $WORK_DIR
cp -r dataset train.py $TMPDIR

cd $TMPDIR
python3 train.py
tar -czf model.tar.gz model

cp -r model model.tar.gz $WORK_DIR
rm -rf $TMPDIR
