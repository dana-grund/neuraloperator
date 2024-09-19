#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=DetErrors
#SBATCH --output=deterministic_errors_correctedEns_new.out
#SBATCH --error=deterministic_errors_correctedEns_new.err
#SBATCH --time=4:0:0

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/ensemble_errors.py