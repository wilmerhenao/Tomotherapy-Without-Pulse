#!/home/wilmer/anaconda3/bin/python

__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import sys
import gurobipy as gp
import numpy as np
import scipy.sparse as sps
import os

def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)

bixels = getvector('data\\Bixels_out.bin', np.int32)
voxels = getvector('data\\Voxels_out.bin', np.int32)
Dijs = getvector('data\\Dijs_out.bin', np.float32)
mask = getvector('data\\optmask.img', np.int32)
print(len(mask))

D = sps.csc_matrix((Dijs,(bixels,voxels)),shape=(14240,65536))
#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(cwd, files)