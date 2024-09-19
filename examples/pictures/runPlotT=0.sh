#!/bin/bash

#SBATCH -A es_schemm_gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --job-name=Plot
#SBATCH --output=plot.out
#SBATCH --error=plot.err
#SBATCH --time=1:0:0

module load gcc/8.2.0 openblas/0.3.20 cuda/11.8.0 cudnn/8.8.1.3 nccl/2.11.4-1 python_gpu/3.11.2
source ~/venv-NO-ens/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/pictures/plotT\=0_ensemble.py
python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/pictures/plotT\=0_ensembleIn.py