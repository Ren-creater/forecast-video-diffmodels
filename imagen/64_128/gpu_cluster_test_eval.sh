#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pritthijit.nath.ml@gmail.com

/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_1e-5
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_1e-4
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_3e-4
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_1k_3e-4
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_rot904_3e-4
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_sep_3e-4
/rds/general/user/zr523/home/venv/bin/python /rds/general/user/zr523/home/researchProject/forecast-diffmodels/imagen/64_128/t07-test-set-evaluation.py -run_name 64_128_rot904_sep_3e-4
