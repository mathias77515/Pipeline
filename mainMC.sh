#!/bin/bash

#SBATCH --job-name=BBPip

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=1        # N
#SBATCH --mem=10G
#SBATCH --time=0-01:00:00
#SBATCH --output=mulitple_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np $SLURM_NTASKS python mainMC.py
