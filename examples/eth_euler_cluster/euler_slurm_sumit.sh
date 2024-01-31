#!/bin/bash

### submit:         > sbatch this_file.sh
### check progess:  > (watch) squeue
### abort:          > scancel JOB_ID

#SBATCH -n 1
#SBATCH --time=00:04:00
#SBATCH --mem-per-cpu=1G
#SBATCH --tmp=1000                        # per node!!
#SBATCH --job-name=FNO
#SBATCH --output=FNO_shear.out
#SBATCH --error=FNO_shear.err
#SBATCH -A es_schemm_gpu

### load environment modules and activate the python venv
source ./euler_setup.sh

### execute any commands we want
SCRIPT_PATH='/cluster/work/climate/dgrund/git/dana-grund/neuraloperator/examples/'
cd $SCRIPT_PATH
pwd
which python
# python plot_FNO_shear.py
python /cluster/work/climate/dgrund/git/dana-grund/neuraloperator/examples/plot_FNO_shear.py

