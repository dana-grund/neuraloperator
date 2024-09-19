#!/bin/bash

#SBATCH --mem-per-cpu=1024G
#SBATCH --ntasks=1
#SBATCH --job-name=SingleEnsembleBaselineCrps
#SBATCH --output=singleEnsembleBaselineCrpsFNO.out
#SBATCH --error=singleEnsembleBaselineCrpsFNO.err
#SBATCH --time=4-0:0:0

module load gcc/8.2.0 openblas/0.3.20 cuda/11.8.0 cudnn/8.8.1.3 nccl/2.11.4-1 python_gpu/3.11.2
source ~/venv-NO-ens/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/baseline_probScores/singleEnsemble_baseline_crps.py