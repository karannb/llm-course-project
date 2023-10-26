#!/bin/sh
#SBATCH --job-name=LLM2
#SBATCH --partition=super
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=llm2.out
#SBATCH --error=llm2.err
#SBATCH --time=1-00:00:00

# activate virtualenv
conda activate mq

# execute
python3 train.py