#!/bin/sh

#SBATCH --time=2:00:00
#SBATCH -p lyceum
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

source activate shuffle

python3 "rename.py" -d "/scratch/jej1g19/datasets/data/valid"
python3 "rename.py" -d "/scratch/jej1g19/datasets/data/train"

python3 "downsample.py" -d "/scratch/jej1g19/datasets/data/valid"
python3 "downsample.py" -d "/scratch/jej1g19/datasets/data/train"
