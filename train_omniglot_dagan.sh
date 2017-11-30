#!/bin/bash

# qsub options:

#$ -l gpu=1

#$ -q gpgpu

# First option informs scheduler that the job requires a gpu.

# Second ensures the job is put in the batch job queue, not the interactive queue



# Set up the CUDA environment

export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-5.1_8.0

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}



export PYTHON_PATH=$PATH

# Activate the relevant virtual environment:

source /home/s1473470/anaconda2/bin/activate deeplearning



#####

# MAKE SURE TO USE THIS, WHATEVER ENVIRONMENT YOU ARE USING.

# This finds the UUID of the GPU with the lowest memory used, and makes sure that is the one you use:

source gpu_lock_script.sh

echo "Chosen gpu:" $CUDA_VISIBLE_DEVICES

#####



# Everything above this point is applicable to all CUDA based packages (Theano, tensorflow, etc).

# The things below might change depending on the environment.

# Set Theano flags and run the job:

python train_omniglot_dagan.py --batch_size 32 --num_of_gpus 1 --experiment_title omniglot_dagan
# End of template

