#!/bin/bash

#SBATCH --job-name=sampling

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=6      # n
#SBATCH --cpus-per-task=3        # N
#SBATCH --mem=20G
#SBATCH --time=0-5:00:00
#SBATCH --output=sampling_r_jobs_%j.log


#mpirun -np $SLURM_NTASKS python main.py
python sampling.py