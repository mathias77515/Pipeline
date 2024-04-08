import numpy as np
import os

folder = 'E2E_nrec2/cmbdust_convTrue_two/maps/'
files = os.listdir(folder)
# E2E_nrec2/cmbdust_convTrue_two/maps/MC_6988219
for i, file in enumerate(files):
    print(i, f'lauching {file}')
    os.system(f'sbatch main.sh {folder + file}')

    
