#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/tn2-sampling-and-evaluation.py -run_name 64_128_rot904_sep_3e-4
