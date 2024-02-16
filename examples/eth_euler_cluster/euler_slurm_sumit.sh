#!/bin/bash

### submit:         > sbatch this_file.sh
### check progess:  > (watch) squeue
### abort:          > scancel JOB_ID

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --time=00:02:00      # hh:m:ss
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=1000           # per node
#SBATCH --job-name=FNO
#SBATCH --output=FNO_shear.out
#SBATCH --error=FNO_shear.err
#SBATCH -A es_schemm_gpu

### load environment modules and activate the python venv
source ./euler_setup.sh

### execute any commands we want
NEURALOP_PATH=/cluster/work/climate/dgrund/git/dana-grund/neuraloperator ### XXX ADAPT HERE
SCRIPT_PATH=${NEURALOP_PATH}/examples
RESULTS_PATH=${NEURALOP_PATH}/examples/plot_FNO_shear

cd $SCRIPT_PATH
CALL='python plot_FNO_shear.py --res 64 -f $RESULTS_PATH --gpu'
echo calling $CALL
echo ------------
$CALL
