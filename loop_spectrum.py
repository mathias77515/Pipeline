import os
import sys

folder = 'E2E_nrec1/maps/'

files = os.listdir(folder)

#print(folder + files[0])
#E2E_nrec1/maps/MC_10046631.pkl
#E2E_nrec2/cmb_convolved_UWB/maps/MC_7723041.pkl
#stop
for i in files:
    #print(folder + i)
	os.system(f'sbatch main.sh {folder + i}')
