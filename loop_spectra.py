import numpy as np
import os

folder = 'E2E_nrec2/cmbdust_convFalse_two/maps/'
files = os.listdir(folder)

for i, file in enumerate(files):
    print(i, f'lauching {file}')
    os.system(f'sbatch main.sh {folder + file}')

    
