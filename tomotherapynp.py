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

def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f,dtype=dtype)
        finally:
            f.close()
    return(data)

bixels = getvector('T:\\Physics Research\\CAP Group\\tools\\OptTool\\dijapp\\dij\\Bixels_out.bin',np.int32)
voxels = getvector('T:\\Physics Research\\CAP Group\\tools\\OptTool\\dijapp\\dij\\Voxels_out.bin',np.int32)
Dijs = getvector('T:\\Physics Research\\CAP Group\\tools\\OptTool\\dijapp\\dij\\Dijs_out.bin',np.float32)

print (bixels.shape,voxels.shape,Dijs.shape)
print (bixels, voxels, Dijs)

D = sps.csc_matrix((Dijs,(bixels,voxels)),shape=(14240,65536))
print(D)
print(D.nnz)
print(np.mean(D*np.ones(14240)))
