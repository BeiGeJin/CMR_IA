#!/bin/bash
echo Please enter a name for your new Anaconda environment:
read ENV_NAME
# conda create -n $ENV_NAME python=3 numpy scipy mkl cython ipykernel
conda create -n $ENV_NAME python=3 numpy scipy ipykernel
source activate $ENV_NAME
pip install mkl cython
python -m ipykernel install --user --name $ENV_NAME --display-name "$ENV_NAME"
python setup.py install
