#!/bin/bash

#SBATCH -A es_schemm_gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --job-name=FNO
#SBATCH --output=FNO_shear_n_train=40000_n_epochs=5-10_correctedEns_gpu.out
#SBATCH --error=FNO_shear_n_train=40000_n_epochs=5-10_correctedEns_gpu.err
#SBATCH --time=2-0:0:0

module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/venv-NO-ens-new/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/plot_FNO_shear_further.py --gpu --ensemble