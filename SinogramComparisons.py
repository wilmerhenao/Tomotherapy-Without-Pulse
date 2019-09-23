__author__ = 'wilmer'
# This one corresponds to the AverageOpeningTime.pdf document (first model)
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import pickle
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
from pylab import Line2D, gca
from scipy.stats import describe
from gurobipy import *
import math
from itertools import product
import pylab as pl
from matplotlib import collections as mc
import itertools

def plotSinogramIndependent(t, L, nameChunk, outputDirectory):
    plt.figure()
    ax = gca()
    lines = []
    for l in range(L):
        for aperture in range(len(t[l])):
            a, b = t[l][aperture]
            lines.append([(a, l), (b, l)])
    lc = mc.LineCollection(lines, linewidths = 3, colors = 'blue')
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    plt.title('Sinogram')
    plt.xlabel('time in seconds')
    plt.ylabel('leaves')
    plt.savefig(outputDirectory + 'SinogramIndependent' + nameChunk + '.png')

nameoutputdirectory = 'outputMultiProj/'
#nameChunk1 = 'pickleresults-ProstatefullModel-MinLOT-0.03-minAvgLot-0.17-vxls-8340-ntnsty-700'
#nameChunk1 = 'pickleresults-ProstatefullModel-MinLOT-0.03-minAvgLot-0.17-vxls-8340-ntnsty-700'
#nameChunk2 = 'pickleresults-ProstatepairModel-MinLOT-0.03-minAvgLot-0.17-vxls-16677-ntnsty-700'
#nameChunk1 = 'pickleresults-Prostate-51-pairModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700'
#nameChunk2 = 'pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700'
nameChunk1 = 'pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-1385-ntnsty-700'
nameChunk2 = 'pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700'
picklefile1 = nameoutputdirectory + nameChunk1 + '.pkl'
picklefile2 = nameoutputdirectory + nameChunk2 + '.pkl'
input = open(picklefile1, 'rb')
sData1 = pickle.load(input)
input = open(picklefile2, 'rb')
sData2 = pickle.load(input)

t1 = sData1['t']
t2 = sData2['t']
L = 64
#plotSinogramIndependent(t1, L, nameChunk1, nameoutputdirectory)
#plotSinogramIndependent(t2, L, nameChunk2, nameoutputdirectory)

#bothOn = [[max(first[0], second[0]), min(first[1], second[1])] for first in t1 for second in t2 if max(first[0], second[0]) <= min(first[1], second[1])]
myeps = 0.001
def range_diff(r1, r2):
    s1, e1 = r1
    s2, e2 = r2
    endpoints = sorted((s1, s2, e1, e2))
    result = []
    if endpoints[0] == s1 and (endpoints[1] - endpoints[0]) > myeps:
        result.append((endpoints[0], endpoints[1]))
    if endpoints[3] == e1 and (endpoints[3] - endpoints[2]) > myeps:
        result.append((endpoints[2], endpoints[3]))
    return result

def multirange_diff(r1_list, r2_list):
    for r2 in r2_list:
        r1_list = list(itertools.chain(*[range_diff(r1, r2) for r1 in r1_list]))
    return r1_list

r1_list = [(1, 1001), (1100, 1201)]
r2_list = [(30, 51), (60, 201), (1150, 1301)]
print(multirange_diff(r1_list, r2_list))

firstOnly = []
secondOnly = []
bothOn = []
for l in range(L):
    firstOnly.append(multirange_diff(t1[l], t2[l]))
    secondOnly.append(multirange_diff(t2[l], t1[l]))
    if len(t1[l]) > 0 and len(t2[l]) > 0:
        bothOn.append([[max(first[0], second[0]), min(first[1], second[1])] for first in t1[l] for second in t2[l] if max(first[0], second[0]) <= min(first[1], second[1])])
    else:
        bothOn.append([])

def plotSinogramIndependentMixed(firstOnly, secondOnly, middleOnly, L, nameChunk1, nameChunk2, outputDirectory):
    plt.figure()
    ax = gca()
    linesFirst = []
    linesSecond = []
    linesMiddle = []
    for l in range(L):
        for aperture in range(len(firstOnly[l])):
            a, b = firstOnly[l][aperture]
            linesFirst.append([(a, l), (b, l)])
        for aperture in range(len(secondOnly[l])):
            a, b = secondOnly[l][aperture]
            linesSecond.append([(a, l), (b, l)])
        for aperture in range(len(middleOnly[l])):
            a, b = middleOnly[l][aperture]
            linesMiddle.append([(a, l), (b, l)])
    lc = mc.LineCollection(linesFirst, linewidths = 3, colors = 'red')
    rc = mc.LineCollection(linesSecond, linewidths = 3, colors = 'blue')
    middlec = mc.LineCollection(linesMiddle, linewidths = 3, colors = 'purple')
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.add_collection(rc)
    ax.add_collection(middlec)
    ax.autoscale()
    #plt.title('Sinogram Comparison of Odd-Even Model vs. Detailed Model')
    plt.title('Sinograms of Low Resolution Model (red) vs. Full Resolution Model (blue)')
    plt.xlabel('time in seconds')
    plt.ylabel('leaves')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True)  #
    ax.set_yticklabels([])
    plt.savefig(outputDirectory + 'Sinogram-Comparison-FullModelvspairModel.pdf', format = 'pdf')

plotSinogramIndependentMixed(firstOnly, secondOnly, bothOn, L, nameChunk1, nameChunk2, nameoutputdirectory)