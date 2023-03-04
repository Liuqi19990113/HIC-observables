#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
srun -n 1 -c 24 python3 ./main.py "$@"
