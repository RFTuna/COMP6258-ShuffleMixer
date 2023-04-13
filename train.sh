#!/bin/sh

#SBATCH --time=0:01:00
#SBATCH -p lyceum
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

python3 "main.py" -r "train" -i "/scratch/jej1g19/datasets/data" -o "/scratch/jej1g19/output"
