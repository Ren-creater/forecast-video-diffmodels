#!/bin/bash
#PBS -l select=1:ncpus=4:mem=48gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00

eval "$(~/anaconda3/bin/conda shell.bash hook)"

source activate research_env

cd /rds/general/user/zr523/home/researchProject/

cd imagen-pytorch
python setup.py develop
cd ..

cd forecast-diffmodels/imagen/64_128

jupyter nbconvert --execute --to notebook 'v_TN3 - Visualizing Metrics.ipynb'