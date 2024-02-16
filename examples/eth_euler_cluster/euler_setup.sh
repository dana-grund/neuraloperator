#!/bin/bash

module load gcc/8.2.0 openblas/0.3.20 cuda/11.8.0 cudnn/8.8.1.3 nccl/2.11.4-1 python_gpu/3.11.2

VENV_DIR=/cluster/home/dgrund/venv-NO-ens
VENV_ACT=${VENV_DIR}/bin/activate

if test -f "$VENV_ACT"; then
   echo "Found the virtual env, activating."
   source $VENV_ACT 
else
   echo "Virtual env not found."
fi
