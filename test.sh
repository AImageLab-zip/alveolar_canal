#!/bin/bash
#SBATCH --partition=students-dev
#SBATCH --gres=gpu:2
#SBATCH --job-name="maxillo-test"

source env/bin/activate
python main.py --config configs/experiment.yaml
