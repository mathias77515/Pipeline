import os
import sys

folder = 'E2E_nrec2/cmbdust_fixAdbicep_DB/maps/'


files = os.listdir(folder)

for i in files:
    #print(folder + i)
	os.system(f'sbatch main.sh {folder + i}')
