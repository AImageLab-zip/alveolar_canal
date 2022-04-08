#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --job-name="maxillo"
#SBATCH --mem=64000
#SBATCH --exclude=aimagelab-srv-00

source env/bin/activate
python main.py --config configs/experiment.yaml
