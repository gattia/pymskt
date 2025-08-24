import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import pymskt.image as image
import pymskt.mesh as mesh
import pymskt.statistics as statistics
import pymskt.utils as utils

RTOL = 1e-4
ATOL = 1e-5
__version__ = "0.1.18"
