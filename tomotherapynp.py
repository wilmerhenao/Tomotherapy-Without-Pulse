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
            data = np.fromfile(f,dtype=dtype)
        finally:
            f.close()
    return(data)

bixels = getvector('data\\Bixels_out.bin', np.int32)
voxels = getvector('data\\Voxels_out.bin', np.int32)
Dijs = getvector('data\\Dijs_out.bin', np.float32)

mask = getvector('optmask.img', np.float32)
print(sum(mask))
# with open('optmask.img', 'rb') as f:
#     block = f.read(512 * 2**10)
#     while block != "":
#         # Do stuff with a block
#         block = f.read(512 * 2**10)

# print (bixels.shape,voxels.shape,Dijs.shape)
# print (bixels, voxels, Dijs)

# D = sps.csc_matrix((Dijs,(bixels,voxels)),shape=(14240,65536))
#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(cwd, files)