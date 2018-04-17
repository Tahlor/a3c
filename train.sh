#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32400M   # memory per CPU core
#SBATCH --gres=gpu:2
#SBATCH --output="train.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

export PATH=~/anaconda3/bin:$PATH
module load anaconda/3/4.3.1
source activate conda3

cd /fslhome/tarch/compute/678/a3c
python3 -u main.py --init "./checkpoints/10"


# To run:
#sbatch ./train.sh

