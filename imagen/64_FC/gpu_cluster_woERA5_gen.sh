#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com

/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Amphan" -horizon 100 -start 0
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Mocha" -horizon 100 -start 0
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Tauktae" -horizon 100 -start 0
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_FC/tw5-forecasting-pipeline.py -region "North Indian Ocean" -name "Maha" -horizon 100 -start 24