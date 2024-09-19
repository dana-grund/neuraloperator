#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --job-name=Plot
#SBATCH --output=plotLastBothDiffVAndU.out
#SBATCH --error=plotLastBothDiffVAndU.err
#SBATCH --time=1:0:0

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/pictures/plot_LastBothAndDiffVAndU.py