#!/bin/bash

#SBATCH -A es_schemm_gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=1
#SBATCH --job-name=FNO
#SBATCH --output=FNO_shear_n_train=10000_n_epochs=2_gpu.out
#SBATCH --error=FNO_shear_n_train=10000_n_epochs=2_gpu.err
#SBATCH --time=3:0:0

module load gcc/8.2.0 openblas/0.3.20 cuda/11.8.0 cudnn/8.8.1.3 nccl/2.11.4-1 python_gpu/3.11.2
source ~/venv-NO-ens/bin/activate

python /cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/plot_FNO_shear.py --gpu