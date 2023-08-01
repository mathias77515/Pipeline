import numpy as np
import yaml

from pipeline import *
from pyoperators import *

if __name__ == "__main__":

    ### Initialization
    pipeline = PipelineEnd2End()

    ### Execution
    pipeline.main()

