#!/bin/bash

#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --job-name=ensembleNiaveCRPS
#SBATCH --output=ensembleNiaveCrpsFNO_corrected.out
#SBATCH --error=ensembleNiaveCrpsFNO_corrected.err
#SBATCH --time=0-3:0:0

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/crps/compute_ensembleCRPS.py