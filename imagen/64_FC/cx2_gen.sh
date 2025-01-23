#!/bin/bash
#PBS -l select=1:ncpus=4:mem=48gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00


eval "$(~/anaconda3/bin/conda shell.bash hook)"

source activate research_env

cd /rds/general/user/zr523/home/researchProject/

cd imagen-pytorch
python setup.py develop
cd ..

# cd forecast-diffmodels/dataproc/

# python md1.py

# cd ../..

cd forecast-diffmodels/imagen/64_FC

python v_t05-forecasting-pipeline.py -region "West Indian Ocean" -name "Emnati" -horizon 100 -start 24
python v_t05-forecasting-pipeline.py -region "North Pacific Ocean" -name "Orlene" -horizon 100 -start 12
python v_t05-forecasting-pipeline.py -region "North Indian Ocean" -name "Mocha" -horizon 100 -start 00
python v_t05-forecasting-pipeline.py -region "North Indian Ocean" -name "Maha" -horizon 100 -start 24
python v_t05-forecasting-pipeline.py -region "Australia" -name "Veronica" -horizon 100 -start 00
python v_t05-forecasting-pipeline.py -region "West Indian Ocean" -name "Gombe" -horizon 100 -start 24
python v_t05-forecasting-pipeline.py -region "North Atlantic Ocean" -name "Ida" -horizon 100 -start 12
python v_t05-forecasting-pipeline.py -region "North Pacific Ocean" -name "Rosyln" -horizon 100 -start 0
python v_t05-forecasting-pipeline.py -region "West Pacific Ocean" -name "Molave" -horizon 100 -start 30

REGION="West Indian Ocean" NAME="Emnati" START=24 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Emnati"
REGION="North Pacific Ocean" NAME="Orlene" START=12 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Orlene"
REGION="North Indian Ocean" NAME="Mocha" START=00 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Mocha"
REGION="North Indian Ocean" NAME="Maha" START=24 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Maha"
REGION="Australia" NAME="Veronica" START=00 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Veronica"
REGION="West Indian Ocean" NAME="Gombe" START=24 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Gombe"
REGION="North Atlantic Ocean" NAME="Ida" START=12 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Ida"
REGION="North Pacific Ocean" NAME="Rosyln" START=0 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Rosyln"
REGION="West Pacific Ocean" NAME="Molave" START=30 jupyter nbconvert --execute --to notebook 'v_T06 - Compare with Forecast Horizon.ipynb' --output "Molave"
