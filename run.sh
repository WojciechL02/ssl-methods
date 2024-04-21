#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate cvenv

python3 main.py --config-path ./configs/config_dae.yaml &
python3 main.py --config-path ./configs/config_mae.yaml &
python3 main.py --config-path ./configs/config_sae.yaml &
wait
