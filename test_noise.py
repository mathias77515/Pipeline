import numpy as np
import yaml

from pipeline import *
from pyoperators import *
from model.externaldata import *
import pysm3
import pysm3.units as u
from pysm3 import utils


with open('params.yml', "r") as stream:
    params = yaml.safe_load(stream)

noise = Noise(params, np.linspace(20, 200, 8))
n = noise._get_noise(143)
sigma = np.std(n, axis=0)
depth = noise._get_depth_from_sigma(sigma)



