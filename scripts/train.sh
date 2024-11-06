#!/bin/sh
#SBATCH --job-name=test
#SBATCH -t 5-00:00:00
#SBATCH -c 20
#SBATCH --gpus=h100-96:2
#SBATCH --mem=900G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tienkhoa@comp.nus.edu.sg

TMPDIR=`mktemp -d`
WORK_DIR=$HOME/coursework/eedi
OUTPUT_DIR=$HOME/coursework/eedi/output_qwen7b_math

cd $WORK_DIR
cp -r dataset train.py $TMPDIR
if [ -d "$OUTPUT_DIR/pretrained_model" ]; then
    cp -r $OUTPUT_DIR/pretrained_model $TMPDIR
fi

cleanup() {
    echo "Copying files back to local directory..."
    mkdir -p $OUTPUT_DIR
    cd $TMPDIR
    tar -czf model.tar.gz model
    cp -r model pretrained_model model.tar.gz $OUTPUT_DIR
    rm -rf $TMPDIR
    echo "Cleanup completed."
}
trap cleanup EXIT

cd $TMPDIR
python3 train.py
