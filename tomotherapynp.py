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
## Function that reads the files produced by Weiguo
def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)

## This function creates a sparse matrix
def sparseWrapper(bixels, voxels, Dijs, mask, totalbeamlets, totalVoxels):
    D = sps.csc_matrix((Dijs, (bixels, voxels)), shape=(totalbeamlets, totalVoxels))
    return(D)

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
beamletsInGantry = 80
## Number of control points (every 2 degrees)
controlPoints = 178
## Total number of beamlets
totalbeamlets = controlPoints * beamletsInGantry
## Dimensions of the 2-Dimensional screen
dimX = 256
dimY = 256
## Total number of voxels in the phantom
totalVoxels = dimX * dimY

bixels = getvector('data\\Bixels_out.bin', np.int32)
voxels = getvector('data\\Voxels_out.bin', np.int32)
Dijs = getvector('data\\Dijs_out.bin', np.float32)
mask = getvector('data\\optmask.img', np.int32)

D = sparseWrapper(bixels, voxels, Dijs, mask, totalbeamlets, totalVoxels)

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(cwd, files)