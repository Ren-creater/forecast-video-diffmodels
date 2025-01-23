#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_FC/m01_64_FC.py -mode execute -epochs 261
