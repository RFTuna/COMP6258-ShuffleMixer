#!/bin/sh

#SBATCH --time=10:00:00
#SBATCH -p lyceum
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

source activate shuffle

python3 "main.py" -r "train" -d "/scratch/jej1g19/datasets/data" -o "/scratch/jej1g19/output"

