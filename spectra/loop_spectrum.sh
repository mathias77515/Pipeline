#!/bin/bash

#SBATCH --job-name=loop_spectrum

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=2      # n
#SBATCH --cpus-per-task=1        # N
#SBATCH --mem=6G
#SBATCH --time=0-1:00:00
#SBATCH --output=mulitple_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#mpirun -np $SLURM_NTASKS python main.py
python loop_spectrum.py