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
import subprocess

# User input goes here and only here
numcores = 11
initialProjections = 51
numberOfLeaves = 80
maxvoxels = 1000 # Never run less than 300
tumorsite = "HelycalGyn"
tumorsite = "Prostate"
timeko = 0.05 # secs
timecl = 0.005 # Minimum LOT
time10 = 10
speed = 24 # degrees per second
t51 = (360/initialProjections) / speed
k10 = math.ceil(time10 / t51)
delta51 = speed * t51 # distance equals speed times time
imrt = False
bigFlag = False
initialTime = time.time()

if 'radiation-math' == socket.gethostname():
    runner277 = "python tomoAverageOnlyTwoProjectionsNeighbors.py "
elif 'IOE-Starchief' == socket.gethostname(): # MY HOUSE
    runner277 = "C:\Intel\python\intelpython3\python tomoAverageOnlyTwoProjectionsNeighbors.py "
elif 'DESKTOP-EA1PG8V' == socket.gethostname(): # MY HOUSE
    runner277 = "C:\Intel\python\intelpython3\python tomoAverageOnlyTwoProjectionsNeighbors.py "
elif ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]): # FLUX
    runner277 = "C:\Intel\python\intelpython3\python tomoAverageOnlyTwoProjectionsNeighbors.py "
else:
    runner277 = "C:\Intel\python\intelpython3\python tomoAverageOnlyTwoProjectionsNeighbors.py "

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
        #self.base_dir = 'data/dij153/prostate/'#153
        # The number of loops to be used in this case
        self.ProjectionsPerLoop = initialProjections
        self.bixelsintween = 1
        self.yBar = 13000
        #self.yBar = 4E11
        self.maxvoxels = maxvoxels
        self.img_filename = 'samplemask.img'
        self.header_filename = 'samplemask.header'
        self.struct_img_filename = 'roimask.img'
        self.struct_img_header = 'roimask.header'
        self.outputDirectory = "output/"
        self.roinames = {}
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.L = numberOfLeaves
        self.get_dim(self.base_dir, 'samplemask.header')
        self.get_totalbeamlets(self.base_dir, 'dij/Size_out.txt')
        self.roimask_reader(self.base_dir, 'roimask.header')
        self.timeko = timeko
        self.timecl = timecl
        self.argumentVariables()
        if 277 == self.maxvoxels:
            bigFlag = False
        else:
            #subprocess.run(runner277 + str(self.timecl) + " " + str(self.timeko) + " 277")
            bigFlag = True
        print('Arguments are changed to: timecl', self.timecl, 'timeko', self.timeko, 'maxvoxels', self.maxvoxels)
        print('Read vectors...')
        self.readWeiguosCase(  )
        self.maskNamesGetter(self.base_dir + self.struct_img_header)
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        # Now remove bixels carefully
        # self.removebixels(self.bixelsintween)
        # Do the smallvoxels again:
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
        if tumorsite == "Prostate":
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 0.001E3 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 1.0E3 #PROSTATE! FOR GYN SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 0.0001E3 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #PROSTATE! FOR GYN SEE BELOW!
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
        self.solveTargetMystery()

    def argumentVariables(self):
        if len(sys.argv) > 1:
            self.timecl = float(sys.argv[1])/1000
        if len(sys.argv) > 2:
            self.timeko = float(sys.argv[2])/100
        if len(sys.argv) > 3:
            self.maxvoxels = int(sys.argv[3])
        print('Arguments were changed to: timecl', self.timecl, 'timeko', self.timeko, 'maxvoxels', self.maxvoxels)

    ## Keep the ROI's in a dictionary
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
                    elif 'PTV' in roitype or 'CTV' in roitype:
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

    def getNumProjections(self):
        with open(self.base_dir + 'motion.txt') as f:
            for i, l in enumerate(f):
                pass
        return i # Do not return -1 because the file has a header.

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
        #print(len(locats))
        self.mask = np.delete(self.mask, locats)
        # intersection of voxels and nonzero
        indices = np.where(np.in1d(self.voxels, locats))[0]
        # Cut whatever is not in the voxels.
        self.bixels = np.delete(self.bixels, indices)
        self.voxels = np.delete(self.voxels, indices)
        self.Dijs = np.delete(self.Dijs, indices)

    def removebixels(self, pitch):
        bixelkill = np.where(0 != (self.bixels % pitch) )
        self.bixels = np.delete(self.bixels, bixelkill)
        self.smallvoxels = np.delete(self.smallvoxels, bixelkill)
        self.Dijs = np.delete(self.Dijs, bixelkill)
        self.mask = self.mask[np.unique(self.smallvoxels)]

    def solveTargetMystery(self):
        import itertools
        import matplotlib.pyplot as plt
        #for t in self.TARGETList
        longMask = [self.mask[v] for v in self.smallvoxels]
        for t in self.TARGETList:
            locats = [i for i, x in enumerate(longMask) if x == t]
            bx = self.bixels[locats]
            vx = self.smallvoxels[locats]
            dx = self.Dijs[locats]
            zx = sorted(zip(vx, bx, dx))
            grouped_data = itertools.groupby(zx, key = lambda x: x[0])
            results = []
            for k, g in grouped_data:
                l = [i[2] for i in list(g)]
                results.append(sum(l)/len(l))
            plt.hist(results, bins=20)  # arguments are passed to np.histogram
            plt.title("Histogram " + str(t))
            plt.savefig(self.outputDirectory + 'Histogram' + str(t) + '.png')
            plt.close()

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        # Assign structures and thresholds for each of them
        if tumorsite == "Prostate":
            self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22]
            self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 78, 10, 10, 10, 10, 10, 10, 10, 10, 1000]
            self.TARGETList = [2]
            self.TARGETThresholds = [78]
        else:
            #self.OARList = [4, 2, 5, 3, 6]
            self.OARList = [4]
            #self.OARThresholds = [1, 1, 1, 1, 1]
            self.OARThresholds = [1]
            self.TARGETList = [8, 7]
            self.TARGETThresholds = [35, 45]
        dtype=np.uint32

        self.bixels = getvector(self.base_dir + 'dij/Bixels_out.bin', np.int32)
        self.voxels = getvector(self.base_dir + 'dij/Voxels_out.bin', np.int32)
        self.Dijs = getvector(self.base_dir + 'dij/Dijs_out.bin', np.float32)
        self.ALLList = self.TARGETList + self.OARList
        # get subsample mask
        img_arr = getvector(self.base_dir + self.img_filename, dtype=dtype)
        img_arr = get_sub_sub_sample(img_arr, self.maxvoxels)
        # get structure file
        struct_img_arr = getvector(self.base_dir + self.struct_img_filename, dtype=dtype)
        # Convert the mask into a list of unitary structures. A voxel gets assigned to only one place
        img_struct = get_structure_mask(reversed(self.ALLList), struct_img_arr)
        # Get the subsampled list of voxels
        self.mask = get_subsampled_mask(img_struct, img_arr)
        # Select only the voxels that exist in the small voxel space provided.
        if tumorsite == "Prostate":
            self.removezeroes([0, 10, 11, 17, 12, 3, 15, 16, 9, 5, 4, 20, 21, 19, 18])
        else:
            self.removezeroes([0])

def solveModel(data):
    bigM = t51
    voxels = range(len(data.mask))
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = k10 + projIni
    projections = range(numProjections)
    projectionsEven = range(0, numProjections - 1, 2)
    leaves = range(data.L)
    leafsD = (data.bixels % data.L).astype(int)
    projectionsD = np.floor(data.bixels / data.L).astype(int)
    m = None
    m = Model("solveModel")
    #m.params.BarConvTol = 1e-6 # Default value is 1e-8
    m.params.MIPGap = 1.0
    m.params.FeasibilityTol = 1e-7 # Guarantee feasibility
    #m.params.InfProofCuts = 0
    #m.params.PumpPasses = 100
    m.params.scaleFlag = 0
    m.params.Threads = 10
    if imrt:
        print("Solving the IMRT-like version of the model")
    else:
        print("Solving the Average LOT Constrained version of the model")
    z = m.addVars(voxels, lb = 0.0, obj = 1.0, vtype = GRB.CONTINUOUS, name = "z")
    t = m.addVars(leaves, projections, obj = 1.0, vtype = GRB.CONTINUOUS, name="t", lb = 0.0, ub = t51)
    z_plus = m.addVars(voxels, lb = 0.0, vtype = GRB.CONTINUOUS, name = "z_plus")
    z_minus = m.addVars(voxels, lb = 0.0, vtype = GRB.CONTINUOUS, name = "z_minus")
    if not imrt:
        s = m.addVars(leaves, projections, vtype=GRB.BINARY, name="s")
        beta = m.addVars(leaves, projections, obj = 1.0, vtype=GRB.BINARY, name="beta")
        mathcalT = m.addVar(vtype=GRB.CONTINUOUS, name="mathcalT", lb=0.0)
        mathcalN = m.addVar(vtype=GRB.INTEGER, name="mathcalN", lb=0)
    if 0:
        oldrun = pickle.load(data.outputDirectory + 'pickleresults-' + str(data.timecl) + '-min-AvgLOT-' + str(data.timeko) + '-voxels277.pkl', 'rb')
        for p, l in list(product(projections, leaves)):
            t[l, p].VarHintVal = oldrun['t_output'][l, p]
            if not imrt:
                s[l, p].VarHintVal = oldrun['s_output'][l, p]
                beta[l, p].VarHintVal = oldrun['beta_output'][l, p]
    m.update()
    hs = [LinExpr(0.0) for _ in voxels]
    [hs[data.smallvoxels[l]].add(data.Dijs[l] * t[leafsD[l], projectionsD[l] + k10]) for l in range(len(data.smallvoxels))]
    [m.addConstr(z[v] == data.yBar * hs[v], name="doses_to_j_yparam[" + str(v) + "]") for v in voxels]
    positive_only = m.addConstrs((z_plus[v] - z_minus[v] == z[v] - data.quadHelperThresh[v] for v in voxels), "positive_only")
    myObj = QuadExpr(0.0)
    for v in voxels:
        myObj.add(data.quadHelperUnder[v] * z_minus[v] * z_minus[v] + data.quadHelperOver[v] * z_plus[v] * z_plus[v])
    if imrt:
        pass
    else:
        closed_start = m.addConstrs((0 == t[l, p] for l in leaves for p in range(k10)), "closed_start")
        fixs = m.addConstrs((0 == s[l, p + 1] for l in leaves for p in projectionsEven), "fixs")
        open_at_all1 = m.addConstrs((data.timecl * (beta[l, p] + beta[l, p + 1]) * s[l, p] / 2.0 <= t[l, p] + t[l, p + 1] for l in leaves for p in projectionsEven), "open_at_all1")
        open_at_all2 = m.addConstrs((data.timecl * (1 - s[l, p]) * beta[l, p] <= t[l, p] for l in leaves for p in projectionsEven), "open_at_all2")
        open_at_all3 = m.addConstrs((data.timecl * (1 - s[l, p]) * beta[l, p + 1] <= t[l, p + 1] for l in leaves for p in projectionsEven), "open_at_all3")
        open_at_all4 = m.addConstrs((s[l, p] <= beta[l, p] for l in leaves for p in projectionsEven), "open_at_all4")
        open_at_all5 = m.addConstrs((s[l, p] <= beta[l, p + 1] for l in leaves for p in projectionsEven), "open_at_all5")
        open_at_all6 = m.addConstrs((t[l, p] <= bigM * beta[l, p] for l in leaves for p in projections), "open_at_all6")
        allts = LinExpr(0.0)
        allns = LinExpr(0.0)
        for l in range(data.L):
            for p in range(k10, numProjections):
                allts.add(t[l,p])
                allns.add(beta[l,p] - s[l, p])
        sumAllOpeningEvents = m.addConstr(mathcalN == allns, "sumAllOpeningEvents")
        sumAllOpeningTimes = m.addConstr(mathcalT == allts, "sumAllOpeningTimes")
        Average_LOT_c = m.addConstr((mathcalT >= data.timeko * mathcalN), "Average_LOT_c")
    m.setObjective(myObj, GRB.MINIMIZE)
    m.update()
    m.optimize()
    valst = np.zeros(numProjections)
    for j in projections:
        for i in range(80):
            valst[j] += t[i,j].x/80
    np.savetxt('t.txt', valst)
    z_output = [v.x for v in m.getVars()[0:len(voxels)]]
    t_output = np.zeros((numProjections, data.L), dtype=float)
    if not imrt:
        s_output = np.zeros((numProjections, data.L), dtype=float)
        beta_output = np.zeros((numProjections, data.L), dtype=float)
    for p, l in list(product(projections, leaves)):
        t_output[p, l] = t[l, p].X
        if not imrt:
            s_output[p, l] = s[l, p].X
            beta_output[p,l] = beta[l, p].X
    tn = np.transpose(np.reshape(t_output, (data.L, numProjections)))
    np.savetxt("foo.csv", tn, delimiter=",")
    if imrt:
        d = {"z_out": z, "z_plus_out": z_plus, "z_minus_out": z_minus, "t_out": t_output, "z_output": z_output}
    else:
        d = {"z_out": z, "z_plus_out": z_plus, "z_minus_out": z_minus, "t_out": t_output, "z_output": z_output, "s_out": s_output, "beta_output": beta_output}
    return(d)

# Plot the dose volume histogram
def plotDVHNoClass(data, z, NameTag='', showPlot=False):
    voxDict = {}
    data.TARGETList = np.intersect1d(np.array(data.TARGETList), np.unique(data.mask))
    data.OARList = np.intersect1d(np.array(data.OARList), np.unique(data.mask))
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
        plt.title('DVH-' + tumorsite + '-IMRT-benchmark')
        print(data.outputDirectory + NameTag + tumorsite + 'IMRTlike.png')
        plt.savefig(data.outputDirectory + NameTag + tumorsite + 'IMRTlike.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.title('DVH-' + tumorsite + 'min. LOT = ' + str(data.timecl) + ' and min.AvgLOT = ' + str(data.timeko))
        print(data.outputDirectory + NameTag + tumorsite + 'minLOT-' + str(data.timecl) + '-min-AvgLOT-' + str(data.timeko)+ '.png')
        plt.savefig(data.outputDirectory + NameTag + tumorsite + 'minLOT-' + str(data.timecl) + '-min-AvgLOT-' + str(data.timeko) + '-vxls' + str(data.maxvoxels) +'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    if showPlot:
        plt.show()
    plt.close()

def plotSinogram(t, L):
    #print('t:', t)
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
    plt.savefig('sinogram.png')

def sinogramAndHistogramNoIMRT(d, data):
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = k10 + projIni
    projectionsEven = range(0, numProjections - 1, 2)
    tim = d["t_out"]
    op = d["s_out"]
    beta = d["beta_output"]
    # contains pairs when the aperture opens and closes
    t = [[]] * data.L # there is one for each of the leaves
    leavelengths = []
    for p in projectionsEven:
        time1 = t51 * p
        time2 = t51 * (p + 1)
        for l in range(data.L):
            if op[p, l]:
                t[l].append([time1 + t51 - tim[p, l], time2 + tim[p, l + 1]])
                leavelengths.append(tim[p, l] + tim[p + 1, l])
            else:
                if tim[p, l] > 0:
                    t[l].append([time1, time1 + tim[p, l]])
                    leavelengths.append(tim[p, l])
                if tim[p + 1, l] > 0:
                    t[l].append([time2, time2 + tim[p + 1, l]])
                    leavelengths.append(tim[p + 1, l])
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
    a = dict()
    a['avLength'] = totalLength / n
    a['minLength'] = minLength
    a['modFactor'] = maxLength / (totalLength/n)
    a['t'] = t
    a['leavelengths'] = leavelengths
    a['s_output'] = op
    a['t_output'] = tim
    a['beta_output'] = beta
    print('average length:', a['avLength'])
    print('minimum length:', minLength)
    print('modulation factor:', a['modFactor'])

    plotSinogram(t, data.L)
    plt.clf()
    binsequence = [i for i in np.arange(min(leavelengths), max(leavelengths), 0.01)] + [max(leavelengths)]
    plt.hist(np.array(leavelengths), bins = binsequence)
    # Add a few extra ticks to the labels
    #extraticks = [a['minLength'], a['avLength']]
    #plt.xticks(list(plt.xticks()[0]) + extraticks)
    plt.xlabel('Leaf Opening Times')
    plt.text(a['minLength'], 7, str(a['minLength'])[0:6], color='r', rotation=89)
    plt.text(a['avLength'], 7, str(a['avLength'])[0:6], color='r', rotation=89)
    plt.title('histogram: Min LOT Goal: ' + str(data.timecl) + ' Actual:' + str(a['minLength'])[0:6] +
              ' AvgLOT goal:' + str(data.timeko) + ' Actual: ' + str(a['avLength'])[0:6] )
    plt.savefig(data.outputDirectory + 'histogram-CL-' + str(data.timecl) + '-min-AvgLOT-' + str(data.timeko) + '-vxls' +
                str(data.maxvoxels) + '.png')
    # Let's pickle save the data results
    output2 = open(data.outputDirectory + 'pickleresults-' + str(data.timecl) + '-min-AvgLOT-' + str(data.timeko) + '-voxels' + str(data.maxvoxels) + '.pkl', 'wb')
    pickle.dump(a, output2)
    output2.close()
    return(t)

dataobject = tomodata()
d = solveModel(dataobject)
plotDVHNoClass(dataobject, d["z_output"], 'dvh')
if imrt:
    pass
else:
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
