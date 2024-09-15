#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=Spectra
#SBATCH --output=plotDist.out
#SBATCH --error=plotDist.err

module load gcc/8.2.0 openblas/0.3.20 cuda/11.8.0 cudnn/8.8.1.3 nccl/2.11.4-1 python_gpu/3.11.2
source ~/venv-NO-ens/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/distribution_plotting/plot_distribution.py