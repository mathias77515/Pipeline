#!/bin/bash

<<<<<<< HEAD
#SBATCH --job-name=FMM
=======
#SBATCH --job-name=BBPip6
>>>>>>> a30ac34a23ac31b1a9458d6eee3ede980e348ea4

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=htc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=1      # n
#SBATCH --cpus-per-task=4        # N
#SBATCH --mem=35G
<<<<<<< HEAD
#SBATCH --time=1-00:00:00
=======
#SBATCH --time=3-00:00:00
>>>>>>> a30ac34a23ac31b1a9458d6eee3ede980e348ea4
#SBATCH --output=mulitple_jobs_%j.log
#SBATCH --array=1-500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
<<<<<<< HEAD
module load mpich
=======

>>>>>>> a30ac34a23ac31b1a9458d6eee3ede980e348ea4
mpirun -np $SLURM_NTASKS python main.py $1