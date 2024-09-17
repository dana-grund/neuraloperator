#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=PlotMeans
#SBATCH --output=MeanEnsDiff.out
#SBATCH --error=MeanEnsDiff.err

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/distribution_plotting/variancees/corrected_ensData/plot_meanEnsDiff_corrected.py
