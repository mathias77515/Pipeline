import os

N = 1

for i in range(0, N):
    os.system(f'sbatch spectrum.sh {i}')