#!/bin/bash

# module load openmpi/4.1.2
module load gcc/8.2.0
module load python/3.10.4

VENV_DIR=/cluster/home/dgrund/venv-NO-ens
VENV_ACT=${VENV_DIR}/bin/activate

if test -f "$VENV_ACT"; then
   echo "Found the virtual env, activating."
   source $VENV_ACT 
else
   echo "Virtual env not found."
fi

echo Using the following python: 
which python 
echo -----
echo