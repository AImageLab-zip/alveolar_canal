#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --account=tesisti
#SBATCH --partition=students-prod
#SBATCH --qos=students-prod
#SBATCH --output=/nas/softechict-nas-2/cmercadante/logs/log.out
#SBATCH --job-name="kwak_unet"

cd /homes/cmercadante/alverolar_canal_3Dtraining
export OMP_NUM_THREADS=1
srun -Q --immediate=10 python kwak_main.py