#!/bin/bash

#SBATCH -A es_schemm_gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --job-name=Plot
#SBATCH --output=plotSeriesRegV.out
#SBATCH --error=plotSeriesRegV.err
#SBATCH --time=1:0:0

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/pictures/plot_seriesRegularV.py