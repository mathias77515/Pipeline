#!/bin/bash

#SBATCH --job-name=fitting

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=4      # n
#SBATCH --cpus-per-task=2        # N
#SBATCH --mem=20G
#SBATCH --time=0-10:00:00
#SBATCH --output=fitting_r_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich
mpirun -np $SLURM_NTASKS python fitting.py
 