#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 5-00:00:00
#SBATCH -c 20
#SBATCH --gpus=h100-47:1
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tienkhoa@comp.nus.edu.sg

TMPDIR=`mktemp -d`
WORK_DIR=$HOME/coursework/eedi

cd $WORK_DIR
cp -r dataset model test.ipynb $TMPDIR

cd $TMPDIR
jupyter nbconvert --to notebook --execute test.ipynb --inplace --debug

cp -r test.ipynb $WORK_DIR
rm -rf $TMPDIR
