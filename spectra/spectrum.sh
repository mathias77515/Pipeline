#!/bin/bash

#SBATCH --job-name=spectrum

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=3      # n
#SBATCH --cpus-per-task=1        # N
#SBATCH --mem=20G
#SBATCH --time=0-10:00:00
#SBATCH --output=spectrum_jobs_%j.log

mpirun -np $SLURM_NTASKS python main.py
#python spectrum.py $1