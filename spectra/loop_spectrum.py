import os

N = 2

for i in range(N):
    os.system(f'sbatch spectrum.sh {i}')