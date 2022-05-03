#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --job-name="maxillo"
#SBATCH --exclude=aimagelab-srv-00
#SBATCH --mem=32G

source env/bin/activate
python main.py --config configs/seg-pretraining.yaml --verbose
