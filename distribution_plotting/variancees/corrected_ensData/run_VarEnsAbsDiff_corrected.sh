#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=Var
#SBATCH --output=AvgVarEnsAbsDiff.out
#SBATCH --error=AvhVarEnsAbsDiff.err

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/distribution_plotting/variancees/corrected_ensData/varEnsAbsDiff_corrected.py
