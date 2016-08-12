#!/home/wilmer/anaconda3/bin/python
__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import gurobipy as grb
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sps
import time
from scipy.stats import describe
import pickle
import matplotlib


def Dsanity():
    p = pickle.load(open('Dsanity.pkl','rb'))
    print(p['D'].transpose())
    print(p['bixels'])
    x = p['D']
    assert(isinstance(x,sps.csr_matrix))
    row, col = x.nonzero()
    tot = np.sort(np.unique(np.unique(col) % 80))
    for i in tot:
        print(i)








Dsanity()