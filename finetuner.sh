#!/bin/sh
#SBATCH --job-name=xl-finetune
#SBATCH --partition=super
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=bfine.out
#SBATCH --error=bfine.err
#SBATCH --time=1-00:00:00

# activate virtualenv
conda activate mq

# execute
python3 finetuner.py
