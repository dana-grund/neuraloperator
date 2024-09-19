#!/bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=Spectra
#SBATCH --output=spectra.out
#SBATCH --error=spectra.err

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/plot_spectra_multi2_1DTotalRegular.py