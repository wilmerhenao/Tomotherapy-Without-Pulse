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
import os

# User input goes here and only here
timeA = 0.077 #secs
timeM = 0.02 #Minimum LOT
time10 = 10
speed = 24 #degrees per second
numcores = 12
initialProjections = 51
numberOfLeaves = 64
do_subsample = False
maxvoxels = 600 # Never run less than 300. Only useful if do_subsample is True
imrt = True
imrtwith20msecondsconstraint = False      #Only active if imrt activated
loadWarmStart = False
relaxedProblem = False
pairSolution = True

# If called externally
executor = ''
if len(sys.argv) > 1:
    executor = sys.argv[0]
    timeM = float(sys.argv[1])
    timeA = float(sys.argv[2])
    # IMRT
    imrt = False
    if int(sys.argv[3]):
        imrt = True
    # IMRT WITH CONSTRAINT
    imrtwith20msecondsconstraint = False
    if int(sys.argv[4]):
        imrtwith20msecondsconstraint = True
    # pair solutions
    pairSolution = False
    if int(sys.argv[5]):
        pairSolution = True
    # relaxed Problem
    relaxedProblem = False
    if int(sys.argv[6]):
        relaxedProblem = True
    loadWarmStart = False
    initialProjections = int(sys.argv[7])
    if int(sys.argv[8]):
        loadWarmStart = True
    if len(sys.argv) > 9:
        maxvoxels = int(sys.argv[9])
        do_subsample = True

# Logical fixes:
if imrtwith20msecondsconstraint and not imrt:
    imrtwith20msecondsconstraint = False
if imrt:
    loadWarmStart = False
    relaxedProblem = False
    pairSolution = False

print('Arguments are: timeM', timeM, 'timeA', timeA, 'maxvoxels', maxvoxels, 'effective?', str(do_subsample), 'imrt', str(imrt),
      'imrtwithConstraint', str(imrtwith20msecondsconstraint), 'relaxedProblem',
      str(relaxedProblem), 'pairSolution', str(pairSolution), 'warmStart', str(loadWarmStart))

tumorsite = "HelycalGyn"
tumorsite = "Prostate"
#tumorsite = "Prostate153"
#tumorsite = "Lung"

if "Lung" == tumorsite or "Prostate153" == tumorsite:
    initialProjections = 153

if "Prostate" == tumorsite or "Prostate153" == tumorsite:
    plotList = [6, 7, 8, 10, 13, 14, 2]

tBAR = (360/initialProjections) / speed
k10 = math.ceil(time10 / tBAR)
delta51 = speed * tBAR #distance equals speed times time
initialTime = time.time()

## Function that reads the files produced by Weiguo
def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)

def get_subsampled_mask(struct_img_mask_full_res, subsampling_img):
    sub_sampled_img_struct = np.zeros_like(struct_img_mask_full_res)
    sub_sampled_img_struct[np.where(subsampling_img)] =  struct_img_mask_full_res[np.where(subsampling_img)]
    return np.copy(sub_sampled_img_struct)

def get_structure_mask(struct_id_list, struct_img_arr):
    img_struct = np.zeros_like(struct_img_arr)
    # Go structure by structure and identify the voxels that belong to it
    for s in struct_id_list:
        img_struct[np.where(struct_img_arr & 2 ** (s - 1))] = s
    return np.copy(img_struct)

## Function that selects roughly the number numelems as a sample. (Usually you get substantially less)
## Say you input numelems=90. Then you get less than 90 voxels in your case.
def get_sub_sub_sample(subsampling_img, numelems):
    sub_sub = np.zeros_like(subsampling_img)
    locations = np.where(subsampling_img)[0]
    #print(locations)
    print('number of elements', len(locations))
    a = np.arange(0,len(locations), int(len(locations)/numelems))
    #print(a)
    sublocations = locations[a]
    sub_sub[sublocations] = 1
    return(sub_sub)

class tomodata:
    ## Initialization of the data
    def __init__(self):
        print('hostname:', socket.gethostname())
        self.base_dir = 'data/dij/HelicalGyn/'
        #self.base_dir = 'data/dij153/HelicalGyn/'
        if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
            self.base_dir = '/scratch/engin_flux/wilmer/dij/HelicalGyn/'
        if tumorsite == "Prostate":
            self.base_dir = 'data/dij/prostate/'  # 51
            if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
                self.base_dir = '/scratch/engin_flux/wilmer/dij/prostate/'
        if tumorsite == "Prostate153":
            self.base_dir = 'data/dij153/prostate/'  # 153
            if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
                self.base_dir = '/scratch/engin_flux/wilmer/dij/prostate/'
        if tumorsite == "Lung":
            self.base_dir = 'data/dij153/lung/'  # 153
            if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
                self.base_dir = '/scratch/engin_flux/wilmer/dij/lung/'
        # The number of loops to be used in this case
        self.ProjectionsPerLoop = initialProjections
        self.bixelsintween = 1
        self.yBar = 39 * 9.33
        self.yBar = 700
        self.maxvoxels = maxvoxels
        self.img_filename = 'samplemask.img'
        self.header_filename = 'samplemask.header'
        self.struct_img_filename = 'roimask.img'
        self.struct_img_header = 'roimask.header'
        self.outputDirectory = "outputMultiProj/"
        self.roinames = {}
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.L = numberOfLeaves
        self.get_dim(self.base_dir, 'samplemask.header')
        self.get_totalbeamlets(self.base_dir, 'dij/Size_out.txt')
        self.roimask_reader(self.base_dir, 'roimask.header')
        self.timeA = timeA
        self.timeM = timeM
        #self.argumentVariables()
        print('Read vectors...')
        self.readWeiguosCase(  )
        self.maskNamesGetter(self.base_dir + self.struct_img_header)
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        #Now remove bixels carefully
        #Do the smallvoxels again:
        _, _, self.smallvoxels, _ = np.unique(self.smallvoxels, return_index=True, return_inverse=True, return_counts=True)
        print('Build sparse matrix.')
        self.totalsmallvoxels = max(self.smallvoxels) + 1 #12648448
        print('totalsmallvoxels:', self.totalsmallvoxels)
        print('a brief description of Dijs array', describe(self.Dijs))
        #self.D = sps.csr_matrix((self.Dijs, (self.smallvoxels, self.bixels)), shape=(self.totalsmallvoxels, self.totalbeamlets))
        self.quadHelperThresh = np.zeros(len(self.mask))
        self.quadHelperUnder = np.zeros(len(self.mask))
        self.quadHelperOver = np.zeros(len(self.mask))
        self.numProjections = self.getNumProjections()
        #######################################
        projIni = 1 + np.floor(max(self.bixels / self.L)).astype(int)
        self.numProjections = k10 + projIni
        self.leafsD = (self.bixels % self.L).astype(int)
        self.projectionsD = np.floor(self.bixels / self.L).astype(int)
        self.bdata = np.zeros((self.numProjections, self.L))
        if tumorsite == "Prostate":
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 15.5 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 2.3 #PROSTATE! FOR GYN SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 5E-3 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #PROSTATE! FOR GYN SEE BELOW!
                    #if 7 == self.mask[i]: # RECTUM!!!
                    #    self.quadHelperOver[i] = 50E-3  # PROSTATE! FOR GYN SEE BELOW!
                    #    self.quadHelperUnder[i] = 0.0  # PROSTATE! FOR GYN SEE BELOW!
                elif 0 == self.mask[i]:
                    print('there is an element in the voxels that is also mask 0')
                self.quadHelperThresh[i] = T
                ########################
        elif "Lung"  == tumorsite:
            print('Gyn Case parameters')
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 0.0001 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 1E1 #GYN! FOR PROSTATE SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 0.003 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #GYN! FOR PROSTATE SEE BELOW!
                elif 0 == self.mask[i]:
                    print('there is an element in the voxels that is also mask 0')
                self.quadHelperThresh[i] = T
                ########################
        else:
            print('Gyn Case parameters')
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 0.0001 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 9E11 #GYN! FOR PROSTATE SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 0.003 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #GYN! FOR PROSTATE SEE BELOW!
                elif 0 == self.mask[i]:
                    print('there is an element in the voxels that is also mask 0')
                self.quadHelperThresh[i] = T
                ########################
        for i in range(len(self.bixels)):
            beamP = self.projectionsD[i] + k10
            beamL = self.leafsD[i]
            if self.bdata[beamP, beamL] < self.Dijs[i]:
                self.bdata[beamP, beamL] = self.Dijs[i]
        # Logging
        self.treatmentName = 'IMRT'
        if imrtwith20msecondsconstraint:
            self.treatmentName += '20msec'
        if not imrt:
            if pairSolution:
                self.treatmentName = 'pairModel'
            else:
                self.treatmentName = 'fullModel'
            if relaxedProblem:
                self.treatmentName += 'relaxedVersion'
        self.chunkName = tumorsite + '-' + str(initialProjections) + '-' + self.treatmentName + '-MinLOT-' + str(timeM) + '-minAvgLot-' + str(timeA) + '-vxls-' + str(self.totalsmallvoxels) + '-ntnsty-' + str(self.yBar)
        self.logfile = ''
        if imrt:
            self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'IMRT.log'
        else:
            if relaxedProblem:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'relaxed.log'
            elif pairSolution:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'pairSolution.log'
            else:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'completeSolution.log'

    ## This function keeps the ROI's in a dictionary
    def maskNamesGetter(self, maskfile):
        lines = tuple(open(maskfile, 'r'))
        for line in lines:
            if 'ROIIndex =' == line[:10]:
                roinumber = line.split(' = ')[1].rstrip()
            elif 'ROIName =' == line[:9]:
                roiname = line.split(' = ')[1].rstrip()
            elif '}' == line[0]:
                self.roinames[roinumber] = roiname
            else:
                pass
                
    ## Takes care of the indices and the mask 
    def roimask_reader(self, base, fname):
        self.OARDict = {}
        self.TARGETDict = {}
        self.SUPPORTDict = {}
        with open(base + fname, 'r') as rmrd:
            for line in rmrd:
                #print(line)
                if 'ROIIndex =' in line:
                    roiidx = int(line.split(' ')[2])
                elif 'ROIName =' in line:
                    roiname = line.split(' ')[2].rstrip()
                elif 'RTROIInterpretedType =' in line:
                    roitype = line.split(' ')[-1]
                    #print('roitype:', roitype)
                    if 'SUPPORT' in roitype or 'AVOIDANCE' in roitype or 'EXTERNAL' in roitype or '=' in roitype:
                        self.SUPPORTDict[roiidx] = roiname
                    elif 'ORGAN' in roitype:
                        self.OARDict[roiidx] = roiname
                    elif 'PTV' in roitype or 'CTV' in roitype or 'TARGET' in roitype:
                        self.TARGETDict[roiidx] = roiname
                    else:
                        print('rio type not defined')
                        pass
                        #sys.exit('ERROR, roi type not defined')
                else:
                    pass
        rmrd.closed
        #Merge all dictionaries
        self.AllDict = dict(self.SUPPORTDict)
        self.AllDict.update(self.OARDict)
        self.AllDict.update(self.TARGETDict)

    ## Get the total number of beamlets
    def get_totalbeamlets(self, base, fname):
        with open(base + fname, 'r') as szout:
            for i, line in enumerate(szout):
                if 1 == i:
                    self.totalbeamlets = int(line)
        szout.closed

    ## Get the dimensions of the voxel big space
    def get_dim(self, base, fname):
        with open(base + fname, 'r') as header:
            dim_xyz = [0] * 3
            for i, line in enumerate(header):
                if 'x_dim' in line:
                    dim_x = int(line.split(' ')[2])
                if 'y_dim' in line:
                    dim_y = int(line.split(' ')[2])
                if 'z_dim' in line:
                    dim_z = int(line.split(' ')[2])
        header.closed
        self.voxelsBigSpace = dim_x * dim_y * dim_z

    ## Create a map from big to small voxel space, the order of elements is preserved but there is a compression to only
    # one element in between.
    def BigToSmallCreator(self):
        # Notice that the order of voxels IS preserved. So (1,2,3,80,7) produces c = (0,1,2,4,3)
        a, b, c, d = np.unique(self.voxels, return_index=True, return_inverse=True, return_counts=True)
        print('BigToSmallCreator:size of c. Size of the problem:', len(c))
        return(c)

    ## Using the motion.txt file provided, this function calculates the number of projections
    # by counting the lines
    def getNumProjections(self):
        with open(self.base_dir + 'motion.txt') as f:
            for i, l in enumerate(f):
                pass
        return i # Do not return -1 because the file has a header.
    
    ## Remove the voxels with a mask of zero (Air Voxels)
    def removezeroes(self, toremove):
        # Next I am removing the voxels that have a mask of zero (0) because they REALLY complicate things otherwise
        # Making the problem larger.
        #-------------------------------------
        # Cut the mask to only the elements contained in the voxel list
        voxelindex = np.zeros_like(self.mask)
        voxelindex[np.unique(self.voxels)] = 1
        self.mask = np.multiply(voxelindex, self.mask)
        locats = np.where(toremove[0] == self.mask)[0]
        if len(toremove) > 1:
            for i in range(1, len(toremove)):
                locats = np.concatenate([locats, np.where(toremove[i] == self.mask)[0]])
        locats.sort()
        self.mask = np.delete(self.mask, locats)
        # intersection of voxels and nonzero
        indices = np.where(np.in1d(self.voxels, locats))[0]
        # Cut whatever is not in the voxels.
        self.bixels = np.delete(self.bixels, indices)
        self.voxels = np.delete(self.voxels, indices)
        self.Dijs = np.delete(self.Dijs, indices)


    ## Deletes the corresponding bixels
    def removebixels(self, pitch):
        bixelkill = np.where(0 != (self.bixels % pitch) )
        bixelkill = np.where(self.bixels < 60)
        self.bixels = np.delete(self.bixels, bixelkill)
        self.smallvoxels = np.delete(self.smallvoxels, bixelkill)
        self.Dijs = np.delete(self.Dijs, bixelkill)
        self.mask = self.mask[np.unique(self.smallvoxels)]

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        # Assign structures and thresholds for each of them in order of how important they are
        if "Prostate" == tumorsite:
            self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22, 10, 11, 17, 12, 3, 15, 16, 9, 5, 4, 20, 21, 19]
            self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 78, 10, 10, 10, 10, 10, 10, 10, 10, 1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            self.TARGETList = [2]
            self.TARGETThresholds = [78]
        elif "Lung" == tumorsite:
            self.OARList = [5, 4, 7, 3, 2, 13, 6, 1]
            self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
            self.TARGETList = [11, 12]
            self.TARGETThresholds = [70, 70]
        else:
            self.OARList = [4]
            self.OARThresholds = [1]
            self.TARGETList = [8, 7]
            self.TARGETThresholds = [35, 45]
        dtype=np.uint32

        self.bixels = getvector(self.base_dir + 'dij/Bixels_out.bin', np.int32)
        self.voxels = getvector(self.base_dir + 'dij/Voxels_out.bin', np.int32)
        self.Dijs = getvector(self.base_dir + 'dij/Dijs_out.bin', np.float32)
        self.ALLList = self.TARGETList + self.OARList
        # get subsample mask (img_arr will have 1 in the positions where there is data)
        img_arr = getvector(self.base_dir + self.img_filename, dtype=dtype)
        # Only use a subsample of img_arr
        if do_subsample:
            img_arr = get_sub_sub_sample(img_arr, self.maxvoxels)
        # get structure file (used for the mask)
        struct_img_arr = getvector(self.base_dir + self.struct_img_filename, dtype=dtype)
        # Convert the mask into a list of unitary structures. A voxel gets assigned to only one place
        img_struct = get_structure_mask(reversed(self.ALLList), struct_img_arr)
        # Get the subsampled list of voxels
        self.mask = get_subsampled_mask(img_struct, img_arr)
        # Select only the voxels that exist in the small voxel space provided.
        if tumorsite == "Prostate":
            self.removezeroes([0, 18])
        else:
            self.removezeroes([0, 10, 14, 15, 8, 16, 9, 17])
            
    ## This function will calculate the maximum bixel to a target coming from a particular beamlet
    def maxTgtDoses(self, numProjections, k10):
        bdoses = np.zeros(self.L, numProjections)

def solveModel(data):
    bigM = tBAR
    voxels = range(len(data.mask))
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = k10 + projIni
    projections = range(numProjections)
    projectionsm1 = range(numProjections - 1)
    projectionsEven = range(0, numProjections - 1, 2)
    leaves = range(data.L)
    leafsD = data.leafsD
    projectionsD = data.projectionsD
    m = None
    m = Model("solveModel")
    m.params.LogFile = data.logFile
    m.params.DisplayInterval = 60
    m.params.TimeLimit = 12 * 3600
    if relaxedProblem:
        m.params.TimeLimit = 2 * 3600
    m.params.partitionPlace = 30
    if imrt:
        m.params.Method = 2
    if not imrt:
        m.params.BarConvTol = 1e-6 # Default value is 1e-8
        m.params.BarQCPConvTol = 1e-6
        m.params.FeasibilityTol = 1e-7 # Guarantee feasibility
        m.params.ImproveStartGap = 5.0
        m.params.ImproveStartTime = 3600 # Change the strategy to finding feasible solution after 1 hour
        m.params.MIPGap = 0.05
        m.params.MIPFocus = 1
        m.params.PreDepRow = 1
        m.params.Presolve = 0  # This is probably giving a better solution at the end
        m.params.ScaleFlag = 2
        m.params.Heuristics = 0.2
        if maxvoxels > 1000:
            m.params.Presolve = 2
            m.params.Method = 0
            m.params.ObjScale = 3
            m.params.SimplexPricing = 3
        if relaxedProblem:
            m.params.Presolve = 2
            m.params.Method = 2
        if maxvoxels > 10000:
            m.params.TimeLimit = 24 * 3600
    if imrt:
        print("Putting together the IMRT-like version of the model")
        m.params.Presolve = -1
    else:
        print('Finding the beamlets that should be closed')
        bdoses = data.maxTgtDoses(numProjections, k10)
        print("Solving the Average LOT Constrained version of the model")
    z = m.addVars(voxels, lb = 0.0, obj = 1.0, vtype = GRB.CONTINUOUS, name = "z")
    t = m.addVars(leaves, projections, obj = 1.0, vtype = GRB.CONTINUOUS, name="t", lb = 0.0, ub = tBAR)
    z_plus = m.addVars(voxels, lb = 0.0, vtype = GRB.CONTINUOUS, name = "z_plus")
    z_minus = m.addVars(voxels, lb = 0.0, vtype = GRB.CONTINUOUS, name = "z_minus")

    ## Preparation of the data for faster speeds

    perPartition = int(numProjections / numcores) + 1
    if imrtwith20msecondsconstraint:
        beta = m.addVars(leaves, projections, obj=1.0, vtype=GRB.BINARY, name="beta", ub=1.0, lb=0.0)
        for p in projections:
            mypartition = int(p/perPartition) + 1
            for l in leaves:
                beta[l, p].Partition = mypartition
                t[l, p].Partition = mypartition
    if not imrt:
        if relaxedProblem:
            variabletype = GRB.CONTINUOUS
            thereisaHint = False # Relaxed problem does not get a hint
        else:
            variabletype = GRB.BINARY
            thereisaHint = True # Real problem gets a hint from the relaxed one.
        beta = m.addVars(leaves, projections, obj = 1.0, vtype=variabletype, name="beta", ub=1.0, lb=0.0)
        if pairSolution:
            gamma = m.addVars(leaves, projections, obj = 1.0, vtype=variabletype, name="gamma", ub=1.0, lb=0.0)
        else:
            blittle = m.addVars(leaves, projections, obj = 1.0, vtype=variabletype, name="blittle", ub=1.0, lb=0.0)
            mlittle = m.addVars(leaves, projections, obj = 1.0, vtype=variabletype, name="mlittle", ub=1.0, lb=0.0)
            elittle = m.addVars(leaves, projections, obj = 1.0, vtype=variabletype, name="elittle", ub=1.0, lb=0.0)
        mathcalT = m.addVar(vtype=GRB.CONTINUOUS, name="mathcalT", lb=0.0)
        mathcalN = m.addVar(vtype=GRB.CONTINUOUS, name="mathcalN", lb=0.0)
        mathcalT.Partition = 0
        mathcalN.Partition = 0
        for p in projections:
            mypartition = int(p/perPartition) + 1
            if pairSolution:
                for l in leaves:
                    beta[l, p].Partition = mypartition
                    gamma[l, p].Partition = mypartition
                    t[l, p].Partition = mypartition
            else:
                for l in leaves:
                    beta[l, p].BranchPriority = 10
                    elittle[l,p].BranchPriority = 5
                    blittle[l,p].VarHintVal = 0
                    # Partition Assignment
                    beta[l, p].Partition = mypartition
                    blittle[l, p].Partition = mypartition
                    mlittle[l, p].Partition = mypartition
                    elittle[l, p].Partition = mypartition
                    t[l, p].Partition = mypartition
        v_actual = 0
        for v in voxels:
            # Find maximum dose to v
            # Find positions where v = voxel in voxelarray in big voxel space
            indices = np.where(data.smallvoxels == v)[0]
            doseCoeffs = data.Dijs[indices]
            # find position(s) of maximum doses in All voxels space
            maxdosesAll = np.where(np.amax(doseCoeffs) == data.Dijs)[0]
            # Find the first one in the intersection of all voxel space and where voxels = this voxel.
            thisbixel = data.bixels[np.intersect1d(maxdosesAll, indices)[0]]
            thisp = np.floor(thisbixel / data.L).astype(int)
            mypartition = int(thisp / perPartition) + 1
            z[v_actual].Partition = mypartition
            z_plus[v_actual].Partition = mypartition
            z_minus[v_actual].Partition = mypartition
            v_actual += 1
        if thereisaHint:
            hintfile = data.outputDirectory + 'hints' + data.chunkName + '.pkl'
            hintfile = hintfile.replace("Model-Min", "ModelrelaxedVersion-Min")
            try:
                myhints = pickle.load(open(hintfile, 'rb'))
            except:
                print('----Running a realexed version of the problem--------------')
                callerstring = 'C:\Intel\python\intelpython3\python multiTool.py ' + str(timeM) + ' ' + str(
                    timeA) + ' ' + str(int(imrt)) + ' ' + str(int(imrtwith20msecondsconstraint)) + ' ' + str(
                    int(pairSolution)) + ' ' + str(1) + ' ' + str(int(loadWarmStart))
                os.system(callerstring)
                myhints = pickle.load(open(hintfile, 'rb'))
                print('---------Finished running the relaxed version of the problem--------------')
            if loadWarmStart:
                wst = pickle.load(open(data.outputDirectory + 'Feasible' + data.chunkName + '.pkl', 'rb'))
            for p, l in list(product(projections, leaves)):
                if loadWarmStart:
                    t[l, p].Start = wst['t_out'][p, l]
                t[l, p].VarHintVal = myhints['t_out'][p, l]; t[l,p].VarHintPri = 10
                if not imrt:
                    if pairSolution:
                        gamma[l, p].VarHintVal = myhints['gamma_out'][p, l];
                        gamma[l, p].VarHintPri = 1
                        beta[l, p].VarHintVal = myhints['beta_output'][p, l];
                        beta[l, p].VarHintPri = 1
                    else:
                        elittle[l, p].VarHintVal = myhints['elittle_out'][p, l]; elittle[l, p].VarHintPri = 1
                        mlittle[l, p].VarHintVal = myhints['mlittle_out'][p, l]; mlittle[l, p].VarHintPri = 1
                        beta[l, p].VarHintVal = myhints['beta_output'][p, l]; beta[l, p].VarHintPri = 2
                    if loadWarmStart:
                        if pairSolution:
                            beta[l, p].Start = wst['beta_out'][p, l]
                            gamma[l, p].Start = wst['gamma_out'][p, l]
                        else:
                            elittle[l, p].Start = wst['elittle_out'][p, l]
                            mlittle[l, p].Start = wst['mlittle_out'][p, l]
                            blittle[l, p].Start = wst['blittle_out'][p, l]
                            beta[l, p].Start = wst['beta_output'][p, l]
            if len(voxels) == len(myhints['z_output']): # Size of the previous run is the same size
                for v in voxels:
                    z[v].VarHintVal = myhints['z_output'][v]; z[v].VarHintPri = 6
                    z_plus[v].VarHintVal = myhints['z_plus_out'][v]; z_plus[v].VarHintPri = 6
                    z_minus[v].VarHintVal = myhints['z_minus_out'][v]; z_minus[v].VarHintPri = 6
                    if loadWarmStart:
                        z[v].Start = wst['z_output'][v]
                        z_plus[v].Start = wst['z_plus_out'][v]
                        z_minus[v].Start = wst['z_minus_out'][v]
            else: # Size of the previous run was different
                zdose = np.zeros(len(voxels))
                for l in range(len(data.smallvoxels)):
                    zdose[data.smallvoxels[l]] += data.Dijs[l] * myhints['t_out'][projectionsD[l] + k10, leafsD[l]]
                zdose *= data.yBar
                for v in voxels:
                    z[v].VarHintVal = zdose[v]; z[v].VarHintPri = 6
                    differenz = zdose[v] - data.quadHelperThresh[v]
                    if differenz >= 0:
                        z_plus[v].VarHintVal = differenz; z_plus[v].VarHintPri = 6
                        z_minus[v].VarHintVal = 0.0; z_minus[v].VarHintPri = 6
                    else:
                        z_minus[v].VarHintVal = -1 * differenz; z_minus[v].VarHintPri = 6
                        z_plus[v].VarHintVal = 0.0; z_plus[v].VarHintPri = 6
    m.update()
    print("Putting together the constraints in the model")
    hs = [LinExpr(0.0) for _ in voxels]
    [hs[data.smallvoxels[l]].add(data.Dijs[l] * t[leafsD[l], projectionsD[l] + k10]) for l in range(len(data.smallvoxels))]
    [m.addConstr(z[v] == data.yBar * hs[v], name="doses_to_j_yparam[" + str(v) + "]") for v in voxels]
    positive_only = m.addConstrs((z_plus[v] - z_minus[v] == z[v] - data.quadHelperThresh[v] for v in voxels), "positive_only")
    myObj = QuadExpr(0.0)
    for v in voxels:
        myObj.add(data.quadHelperUnder[v] * z_minus[v] * z_minus[v] + data.quadHelperOver[v] * z_plus[v] * z_plus[v])
    closed_ghost = m.addConstrs((0 == t[l, p] for l in leaves for p in range(k10)), "closed_ghost")
    close_zeros = list()
    czcounter = 0
    for l in leaves:
        for p in projections:
            if data.bdata[p, l] < 0.0001:
                close_zeros.append(m.addConstr((0 == t[l, p]), 'close_zeros_' + str(l) + '_' + str(p)))
                czcounter += 1
    print('closing a total beamlets of:', czcounter)
    if imrt:
        if imrtwith20msecondsconstraint:
            # Force the creation of
            twentymsecond_a = m.addConstrs((0.02 * beta[l, p] <= t[l, p] for l in leaves for p in projections), "fivesecond_a")
            twentymsecond_b = m.addConstrs((t[l, p] <= tBAR * beta[l, p] for l in leaves for p in projections),
                                        "fivesecond_b")
    else:
        close_ghost_beta = m.addConstrs((0 == beta[l, p] for l in leaves for p in range(k10)), "close_ghost_beta")
        time_per_projection_b = m.addConstrs((t[l, p] <= tBAR * beta[l, p] for l in leaves for p in projections),
                                             "time_per_projection_b")
        allts = LinExpr(0.0)
        allns = LinExpr(0.0)
        if pairSolution:
            cancel_odd = m.addConstrs(
                (0 == gamma[l, p + 1] for l in leaves for p in projectionsEven),
                "cancel_odd")
            gamma_1 = m.addConstrs(
                (gamma[l, p] <= beta[l, p] + beta[l, p + 1] for l in leaves for p in projectionsEven),
                "gamma_1")
            gamma_2 = m.addConstrs(
                (beta[l, p] + beta[l, p + 1] <= 2 * gamma[l, p] for l in leaves for p in projectionsEven),
                "gamma_2")
            minimum_lot = m.addConstrs(
                (t[l, p] + t[l, p + 1] >= data.timeM * gamma[l, p] for l in leaves for p in projectionsEven),
                "minimum_lot")
            for l in range(data.L):
                for p in range(k10, numProjections):
                    allts.add(t[l,p])
                    allns.add(gamma[l, p])
        else:
            time_per_projection_a = m.addConstrs((tBAR * mlittle[l, p] <= t[l, p] for l in leaves for p in projections), "time_per_projection_a")
            three_options = m.addConstrs((elittle[l, p] + mlittle[l, p] + blittle[l, p] == beta[l, p] for l in leaves for p in projections), "three_options")
            m_follows_m_or_e = m.addConstrs((mlittle[l, p + 1] <= mlittle[l, p] + elittle[l, p] for l in leaves for p in projectionsm1),"m_follows_m_or_e")
            b_follows_m_or_e = m.addConstrs(
                (blittle[l, p + 1] <= mlittle[l, p] + elittle[l, p] for l in leaves for p in projectionsm1),
                "b_follows_m_or_e")
            minimul_lot_eb = m.addConstrs((t[l, p] + t[l, p + 1] >= data.timeM * (elittle[l, p] + blittle[l, p + 1] - 1) for l in leaves for p in projectionsm1), "minimum_lot_eb")
            minimul_lot_e = m.addConstrs((t[l, p] >= data.timeM * (elittle[l, p] + elittle[l, p + 1] - beta[l, p + 1]) for l in leaves for p in projectionsm1), "minimum_lot_e")
            for l in range(data.L):
                for p in range(k10, numProjections):
                    allts.add(t[l,p])
                    allns.add(elittle[l, p])
        sumAllOpeningEvents = m.addConstr(mathcalN == allns, "sumAllOpeningEvents")
        sumAllOpeningTimes = m.addConstr(mathcalT == allts, "sumAllOpeningTimes")
        Average_LOT_c = m.addConstr((mathcalT >= data.timeA * mathcalN), "Average_LOT_c")
    m.setObjective(myObj, GRB.MINIMIZE)
    m.update()
    print('--- Starting the GUROBI optimization ---')
    m.write('pairs.mps')
    m.optimize()
    m.printQuality()
    z_output = [v.x for v in m.getVars()[0:len(voxels)]]
    zplus_output = np.zeros(len(z_output), dtype=float)
    zminus_output = np.zeros(len(z_output), dtype=float)
    t_output = np.zeros((numProjections, data.L), dtype=float)
    if not imrt:
        beta_output = np.zeros((numProjections, data.L), dtype=float)
        if pairSolution:
            gamma_output = np.zeros((numProjections, data.L), dtype=float)
        else:
            blittle_output = np.zeros((numProjections, data.L), dtype=float)
            mlittle_output = np.zeros((numProjections, data.L), dtype=float)
            elittle_output = np.zeros((numProjections, data.L), dtype=float)
    for v in range(len(z_output)):
        zminus_output[v] = z_minus[v].x
        zplus_output[v] = z_plus[v].x
    for p, l in list(product(projections, leaves)):
        t_output[p, l] = t[l, p].x
        if not imrt:
            beta_output[p,l] = beta[l, p].X
            if pairSolution:
                gamma_output[p, l] = gamma[l, p].X
            else:
                blittle_output[p, l] = blittle[l, p].x
                mlittle_output[p, l] = mlittle[l, p].x
                elittle_output[p, l] = elittle[l, p].x
    #tn = np.transpose(np.reshape(t_output, (data.L, numProjections)))
    #np.savetxt("foo.csv", tn, delimiter=",")
    if imrt:
        d = {"z_out": z, "z_plus_out": z_plus, "z_minus_out": z_minus, "t_out": t_output, "z_output": z_output,
             "objVal": m.objVal}
    else:
        if pairSolution:
            d = {"t_out": t_output, "z_output": z_output, "z_plus_out": zplus_output, "z_minus_out": zminus_output,
                 "gamma_out": gamma_output, "beta_output": beta_output,
                 "slackAvgLOT": Average_LOT_c.getAttr("Slack"),
                 "gurobisT": mathcalT.x, "gurobisN": mathcalN.x, "objVal": m.objVal}
        else:
            d = {"t_out": t_output, "z_output": z_output, "z_plus_out": zplus_output, "z_minus_out": zminus_output,
                 "mlittle_out": mlittle_output, "blittle_out": blittle_output,
                 "elittle_out": elittle_output, "beta_output": beta_output, "slackAvgLOT": Average_LOT_c.getAttr("Slack"),
                 "gurobisT": mathcalT.x, "gurobisN": mathcalN.x, "objVal": m.objVal}
        if relaxedProblem:
            outputFile = open(data.outputDirectory + 'hints' + data.chunkName + '.pkl', 'wb')
        else:
            outputFile = open(data.outputDirectory + 'Feasible' + data.chunkName + '.pkl', 'wb')
        pickle.dump(d, outputFile)
        outputFile.close()
    print(m.params)
    return(d)

# Plot the dose volume histogram
def plotDVHNoClass(data, z, NameTag='', showPlot=False):
    voxDict = {}
    data.TARGETList = np.intersect1d(np.array(data.TARGETList), np.unique(data.mask))
    data.TARGETList = np.intersect1d(np.array(data.TARGETList), plotList)
    data.OARList = np.intersect1d(np.array(data.OARList), np.unique(data.mask))
    data.OARList = np.intersect1d(np.array(data.OARList), plotList)
    for t in data.TARGETList:
        voxDict[t] = np.where(data.mask == t)[0]
    for o in data.OARList:
        voxDict[o] = np.where(data.mask == o)[0]
    dose = np.array([z[j] for j in range(data.totalsmallvoxels)])
    plt.clf()
    for index, sValues in voxDict.items():
        sVoxels = sValues
        hist, bins = np.histogram(dose[sVoxels], bins=100)
        dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
        dvh = np.insert(dvh, 0, 1)
        plt.plot(bins, dvh, label=data.AllDict[index], linewidth=2)
    lgd = plt.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    if imrt:
        if imrtwith20msecondsconstraint:
            plt.title('DVH-' + tumorsite + ' IMRT benchmark with 20 msec constraint')
        else:
            plt.title('DVH-' + tumorsite + ' IMRT benchmark')
    else:
        plt.title('DVH-' + tumorsite + 'min. LOT = ' + str(data.timeM) + ' and min.AvgLOT = ' + str(data.timeA))
    print(data.outputDirectory + 'DVH' + data.chunkName + '.png')
    plt.savefig(data.outputDirectory + 'DVH' + data.chunkName + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    if showPlot:
        plt.show()
    plt.close()

def plotSinogram(t, L, data):
    plt.figure()
    ax = gca()
    lines = []
    for l in range(L):
        for aperture in range(len(t[l])):
            a, b = t[l][aperture]
            lines.append([(a, l), (b, l)])
    lc = mc.LineCollection(lines, linewidths = 3, colors = 'red')
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    plt.title('Sinogram')
    plt.xlabel('time in seconds')
    plt.ylabel('leaves')
    plt.savefig(data.outputDirectory + 'Sinogram' + data.chunkName + '.png')

def calculateTIMRT(numProjections, tBAR, tim, data):
    # t is a list that contains a list per leaf. In this inner list, you get the time when it opens and the time when it
    # closes as a pair. leavelengths will contain the total opening times
    t = [list() for _ in range(data.L)] # there is one for each of the leaves
    leavelengths = []
    projectionsEven = range(0, numProjections - 1, 2)
    for p in projectionsEven:
        time1 = tBAR * p
        time2 = tBAR * (p + 1)
        for l in range(data.L):
            if tim[p, l] > 0:
                t[l].append([time1, time1 + tim[p, l]])
                leavelengths.append(tim[p, l])
            if tim[p + 1, l] > 0:
                t[l].append([time2, time2 + tim[p + 1, l]])
                leavelengths.append(tim[p + 1, l])
    return([t, leavelengths])

def calculateT(numProjections, tBAR, tim, data, elittle, mlittle, blittle):
    # t is a list that contains a list per leaf. In this inner list, you get the time when it opens and the time when it
    # closes as a pair. leavelengths will contain the total opening times

    t = [list() for _ in range(data.L)] # there is one for each of the leaves
    projections = range(0, numProjections)
    leavelengths = []
    for l in range(data.L):
        # This implies that the opening is new or continuing from before
        continuousopening = False
        for p in projections:
            timeright = tBAR * (p + 1)
            timeleft = tBAR * p
            # Handle special case of last projection
            if p == numProjections - 1:
                if continuousopening:
                    endingtime = timeleft + tim[p, l]
                    t[l].append([beginningtime, endingtime])
                    leavelengths.append(endingtime - beginningtime)
                    continuousopening = False
                else:
                    if 1 == elittle[p, l]:
                        # If it opened in the right projection it HAS to be centered
                        midtime = timeleft + tBAR / 2.0
                        beginningtime = midtime - tim[p, l] / 2.0
                        endingtime = midtime + tim[p, l] / 2.0
                        t[l].append([beginningtime, endingtime])
                        leavelengths.append(endingtime - beginningtime)
                        continuousopening = False  # this line unnecessary
            else:
                if continuousopening:
                    if (1 == blittle[p, l] or (0 == mlittle[p + 1, l] + blittle[p + 1, l])):
                        # This is when a continuing aperture closes in this projection
                        if 1 == blittle[p, l]:
                            endingtime = timeleft + tim[p, l]
                        else:
                            endingtime = timeright
                        t[l].append([beginningtime, endingtime])
                        leavelengths.append(endingtime - beginningtime)
                        continuousopening = False
                    else:
                        # The aperture continues
                        pass
                else:
                    if elittle[p, l] == 1 and tim[p, l] > 0:
                        # A new opening event started.
                        if 0 == mlittle[p + 1, l] + blittle[p + 1, l] or p == (numProjections - 1):
                            # If it closed right now it HAS to be centered
                            midtime = timeleft + tBAR / 2.0
                            beginningtime = midtime - tim[p, l] / 2.0
                            endingtime = midtime + tim[p, l] / 2.0
                            t[l].append([beginningtime, endingtime])
                            leavelengths.append(endingtime - beginningtime)
                            continuousopening = False #this line unnecessary
                        # But if didn't close it must continue
                        else:
                            beginningtime = timeright - tim[p, l]
                            # endingtime, leavelengths and t not assigned
                            continuousopening = True
                    else:
                        #Leaves were closed and they continue to be closed. Nothing happened
                        pass
    return([t, leavelengths])



def sinogramAndHistogramYesIMRT(d, data):
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = k10 + projIni
    projections = range(numProjections)
    tim = d["t_out"]

    abc = dict()
    abc['numProjections'] = numProjections
    abc['tBAR'] = tBAR
    abc['tim'] = tim
    abc['data'] = data
    with open(data.outputDirectory + 'calculateT' + data.chunkName + '.pkl', "wb") as f:
        pickle.dump(abc, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    t, leavelengths = calculateTIMRT(numProjections, tBAR, tim, data)
    plotSinogram(t, data.L, data)
    plt.clf()
    binsequence = [i for i in np.arange(min(leavelengths), max(leavelengths), 0.01)] + [max(leavelengths)]
    plt.hist(np.array(leavelengths), bins = binsequence)
    # Add a few extra ticks to the labels
    #extraticks = [a['minLength'], a['avLength']]
    #plt.xticks(list(plt.xticks()[0]) + extraticks)
    plt.xlabel('Leaf Opening Times')
    if imrtwith20msecondsconstraint:
        plt.title('histogram IMRT with intensity: ' + str(data.yBar) + 'with cutoff 20 msecs')
    else:
        plt.title('histogram IMRT with intensity: ' + str(data.yBar))
    plt.savefig(data.outputDirectory + 'histogram' + data.chunkName  + '.png')
    abc = dict()
    totalLength = 0.0
    n = 0
    minLength = 1000.0
    maxLength = -1.0
    for l in range(data.L):
        for aperture in range(len(t[l])):
            a, b  = t[l][aperture]
            totalLength += b - a
            maxLength = max(maxLength, b-a)
            n += 1
            # min length
            if b-a > 0.000001:
                minLength = min(b-a, minLength)
    abc['avLength'] = totalLength / n
    abc['totalLength'] = totalLength
    abc['minLength'] = minLength
    abc['modFactor'] = maxLength / (totalLength / n)
    abc['t'] = t
    abc['leavelengths'] = leavelengths
    abc['objVal'] = d['objVal']
    print('average length measured by me:', abc['avLength'])
    print('objective Value:', abc['objVal'])
    output3 = open(data.outputDirectory + 'pickleresults-' + data.chunkName + '.pkl', 'wb')
    pickle.dump(abc, output3, pickle.HIGHEST_PROTOCOL)
    output3.close()

def calculateTpairSolution(numProjections, tBAR, tim, data, gamma):
    # t is a list that contains a list per leaf. In this inner list, you get the time when it opens and the time when it
    # closes as a pair. leavelengths will contain the total opening times
    t = [list() for _ in range(data.L)] # there is one for each of the leaves
    projections = range(0, numProjections)
    projectionsEven = range(0, numProjections, 2)
    leavelengths = []
    for l in range(data.L):
        # This implies that the opening is new or continuing from before
        for p in projectionsEven:
            timemiddle = tBAR * (p + 1)
            if p == numProjections - 1: # last projections
                if gamma[p, l]:
                    beginningtime = tBAR * p; endingtime = beginningtime + t[p, l]
                    t[l].append([beginningtime, endingtime])
                    leavelengths.append(endingtime - beginningtime)
            if gamma[p, l]:
                beginningtime = timemiddle - tim[p, l]; endingtime = timemiddle + tim[p + 1, l]
                t[l].append([beginningtime, endingtime])
                leavelengths.append(endingtime - beginningtime)
    return([t, leavelengths])

def sinogramAndHistogramNoIMRT(d, data):
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = k10 + projIni
    tim = d["t_out"]
    beta = d["beta_output"]
    abc = dict()
    if pairSolution:
        gamma = d["gamma_out"]
        abc['gamma'] = gamma
    else:
        elittle = d["elittle_out"]
        blittle = d["blittle_out"]
        mlittle = d["mlittle_out"]
        abc['elittle'] = elittle
        abc['mlittle'] = mlittle
        abc['blittle'] = blittle

    abc['numProjections'] = numProjections
    abc['tBAR'] = tBAR
    abc['tim'] = tim
    abc['data'] = data
    abc['z_output'] = d['z_output']

    with open(data.outputDirectory + 'calculateT' + data.chunkName + '.pkl', "wb") as f:
        pickle.dump(abc, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # contains pairs when the aperture opens and closes
    if pairSolution:
        t, leavelengths = calculateTpairSolution(numProjections, tBAR, tim, data, gamma)
    else:
        t, leavelengths = calculateT(numProjections, tBAR, tim, data, elittle, mlittle, blittle)
    abc = dict()
    totalLength = 0.0
    n = 0
    minLength = 1000.0
    maxLength = -1.0
    for l in range(data.L):
        for aperture in range(len(t[l])):
            a, b  = t[l][aperture]
            totalLength += b - a
            #print('mynumbers', b - a)
            maxLength = max(maxLength, b-a)
            n += 1
            # min length
            if b-a > 0.000001:
                minLength = min(b-a, minLength)
    abc['avLength'] = totalLength / n
    abc['totalLength'] = totalLength
    abc['minLength'] = minLength
    abc['modFactor'] = maxLength / (totalLength / n)
    abc['t'] = t
    abc['leavelengths'] = leavelengths
    abc['t_output'] = tim
    abc['beta_output'] = beta
    abc['slackAvgLOT'] = d["slackAvgLOT"]
    abc['gurobisT'] = d['gurobisT']
    abc['myN'] = len(leavelengths)
    abc['mySecondN'] = n
    abc['gurobisN'] = d['gurobisN']
    abc['gurobiAvLength'] = abc['gurobisT'] / abc['gurobisN']
    abc['objVal'] = d['objVal']
    print('whole a package:', abc)
    print('average length measured by me:', abc['avLength'], ' and measured by GUROBI: ', abc['gurobiAvLength'])
    print('objective Value:', abc['objVal'])
    print('minimum length:', minLength)
    print('modulation factor:', abc['modFactor'])
    plotSinogram(t, data.L, data)
    plt.clf()
    binsequence = [i for i in np.arange(min(leavelengths), max(leavelengths), 0.01)] + [max(leavelengths)]
    plt.hist(np.array(leavelengths), bins = binsequence)
    # Add a few extra ticks to the labels
    plt.xlabel('Leaf Opening Times')
    plt.text(abc['minLength'], 7, str(abc['minLength'])[0:6], color='r', rotation=89)
    plt.text(abc['avLength'], 7, str(abc['avLength'])[0:6], color='r', rotation=89)
    plt.title('histogram: Min LOT Goal: ' + str(data.timeM) + ' Actual:' + str(abc['minLength'])[0:6] +
              ' AvgLOT goal:' + str(data.timeA) + ' Actual: ' + str(abc['avLength'])[0:6] )
    plt.savefig(data.outputDirectory + 'histogram' + data.chunkName + '.png')
    # Let's pickle save the data results
    output2 = open(data.outputDirectory + 'pickleresults-' + data.chunkName + '.pkl', 'wb')
    pickle.dump(abc, output2)
    output2.close()
    return(t)

dataobject = tomodata()
d = solveModel(dataobject)
# Save info to create dvhs later
#####################################
#####################################
#####################################
output2 = open(dataobject.outputDirectory + dataobject.chunkName + '-z.pkl', 'wb')
pickle.dump(d["z_output"], output2)
output2.close()
output = open(dataobject.outputDirectory + dataobject.chunkName + '-dataobject.pkl', 'wb')
pickle.dump(dataobject, output)
output.close()
#####################################
#####################################
#####################################
plotDVHNoClass(dataobject, d["z_output"], 'dvh')
if imrt:
    sinogramAndHistogramYesIMRT(d, dataobject)
else:
    if not relaxedProblem:
        t = sinogramAndHistogramNoIMRT(d, dataobject)

print('total time:', time.time() - initialTime)
sys.exit()

start_time = time.time()
# Erase everything in the warmstart file
f = open("warmstart.dat", "w")
f.close()
oldobj = np.inf
for i in [1, 2, 4, 8, 16, 32]:
    start_time = time.time()
    #pstring = runAMPL(maxvoxels, i, tumorsite)
    totalAMPLtime = (time.time() - start_time)
    print("--- %s seconds running the AMPL part---" % totalAMPLtime)
    z, betas, B, cgamma, lgamma, newobj = readDosefromtext(pstring)
    print('new obj:', newobj)
    mej = (newobj - oldobj)/oldobj
    oldobj = newobj
    print('reduction:', mej)
    if mej < 0.01:
        break

output2 = open('z.pkl', 'wb')
pickle.dump(z, output2)
output2.close()
output = open('dataobject.pkl', 'wb')
try:
    pickle.dump(dataobject, output)
except:
    print("dataobject was never defined")
output.close()
totalAMPLtime = (time.time() - start_time)
print("--- %s seconds running the AMPL part---" % totalAMPLtime)
if len(sys.argv) > 4:
    print("tabledresults: ", sys.argv[1], sys.argv[2], sys.argv[3], dataobject.totalsmallvoxels, totalAMPLtime)
# Ignore errors that correspond to DVH Plot
try:
    pass
    # Correct this and bring back the plotting when I can.
    #plotDVHNoClass(dataobject, z, 'dvh')
except IndexError:
    print("Index is out of bounds and no DVH plot will be generated. However, I am ignoring this error for now.")
# Output ampl results for the next run in case something fails.
text_output = open("amploutput.txt", "wb")
text_output.write(pstring)
text_output.close()
