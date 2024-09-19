#!/bin/bash

#SBATCH -A es_schemm_gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=Move
#SBATCH --output=moving.out
#SBATCH --error=moving.err

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/move_checkpoint.py