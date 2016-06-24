#!/home/wilmer/anaconda3/bin/python

__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

from data import *
import gurobipy as grb
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sps
import os

## Class definition of the gurobi object that handles creation and execution of the model
# Original template from Troy Long.
class tomotherapyNP(object):
    def __init__(self, datastructure):
        print('Reading in data...')
        self.data = datastructure
        print('done')
        print('Constructing Gurobi model object...')
        self.mod = grb.Model()
        print('done')
        print('Building main decision variables (dose, binaries)...')
        self.buildVariables()
        print('Main variables and dose done')

    ## Wrapper for the optimize function
    def solveModel(self):
        self.mod.optimize()

    ## This function builds variables to be included in the model.
    def buildVariables(self):
        # Addition of t variables. All terms according to the terminology in the writeup.
        ## IntensityVariable
        self.yVar = self.mod.addVar(lb = 0.0, ub = self.data.maxIntensity, obj=0.0, vtype=grb.GRB.CONTINUOUS,
                                    name="intensity", column=None)
        ## Time variable
        self.timeVars = [None] * (self.data.N * self.data.K)
        ## Binary Variable. I call it delta in the writeup
        self.BinaryVars = [None] * ((self.data.N * self.data.K) + self.data.N)
        ## xi Variables. Helper variables to create a continuous binary variable
        self.xiVars = [None] * (self.data.N * self.data.K)
        ## zeta Variables. Helper variables to create a continuous binary variable
        self.zetaVars = [None] * (self.data.N * self.data.K)
        self.zeeplusVars = [None] * (self.data.N * self.data.K)
        self.zeeminusVars = [None] * (self.data.N * self.data.K)

        for k in range(0, self.data.K):
            for i in range(0, self.data.N):
                self.timeVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=1.0, obj=0.0, vtype=grb.GRB.CONTINUOUS,
                                                                     name="t_{" + str(i) + "," + str(k) + "}",
                                                                     column=None)
                self.xiVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=0.0,
                                                                   vtype=grb.GRB.CONTINUOUS,
                                                                   name="xi_{" + str(i) + "," + str(k) + "}",
                                                                   column=None)
                self.zetaVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=0.0,
                                                                     vtype=grb.GRB.CONTINUOUS,
                                                                     name="zeta_{" + str(i) + "," + str(k) + "}",
                                                                     column=None)
                self.gammaplusVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=0.0,
                                                                     vtype=grb.GRB.CONTINUOUS,
                                                                     name="gammaplus_{" + str(i) + "," + str(k) + "}",
                                                                     column=None)
                self.gammaminusVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=0.0,
                                                                     vtype=grb.GRB.CONTINUOUS,
                                                                     name="gammaminus_{" + str(i) + "," + str(k) + "}",
                                                                     column=None)
                self.binaryVars[i + k * self.data.N] = self.mod.addVar(lb = 0.0, ub=1.0, obj=0.0, vtype=grb.GRB.BINARY,
                                                                       name="binary_{" + str(i) + "," + str(k) + "}",
                                                                       column=None)
        ## Initialize extra members of binaryvars that are needed. Since there's an extra edge (see writeup)
        for i in range(0, self.data.N):
            self.binaryVars[i + self.data.K * self.data.N] = self.mod.addVar(lb=0.0, ub=1.0, obj=0.0,
                                                                                    vtype=grb.GRB.BINARY,
                                                                                    name="binary_{" + str(i) +
                                                                                         ", extramember}",
                                                                                    column=None)

        ## This is the variable that will appear in the $z_{j}$ constraint. One per actual voxel in small space.
        self.zeeVars = [None] * (self.data.smallvoxelspace)
        for i in range(0, self.data.smallvoxelspace):
            self.zeeVars[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=1.0, vtype=grb.GRB.CONTINUOUS,
                                                                     name="zee_{" + str(i) + "}",
                                                                     column=None)
        # Lazy update of gurobi
        self.mod.update()

        ## Add some constraints
        self.absoluteValueRemovalConstraint = [None] * (self.data.N * self.data.K)
        self.xiconstraint1 = [None] * (self.data.N * self.data.K)
        self.xiconstraint2 = [None] * (self.data.N * self.data.K)
        self.xiconstraint3 = [None] * (self.data.N * self.data.K)
        for k in range(0, self.data.K):
            for i in range(0, self.data.N):
                self.absoluteValueRemovalConstraint[i + k * self.data.N] = self.mod.addConstr(
                    self.binaryVars[i + (k + 1) * self.data.N] - self.binaryVars[i + k * self.data.N],
                    grb.GRB.EQUAL,
                    self.gammaplus[i + k * self.data.N] - self.gammaminus[i + k * self.data.N],
                    name="rmabs_{" + str(i) + "," + str(k) + "}")
                self.xiconstraint1[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.data.fractionUB * self.binaryVars[i + k * self.data.N],
                    name="xiconstraint1_{" + str(i) + "," + str(k) + "}")
                self.xiconstraint2[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.timeVars[i + k * self.data.N],
                    name="xiconstraint2_{" + str(i) + "," + str(k) + "}")
                self.xiconstraint3[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.GREATER_EQUAL,
                    self.timeVars[i + k * self.data.N] - (1 - self.binaryVars[i + k * self.data.N]) * self.data.fractionUB,
                    name="xiconstraint3_{" + str(i) + "," + str(k) + "}")
                self.zetaconstraint1[i + k * self.data.N] = self.mod.addConstr(
                    self.zetaVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.data.fractionUB * self.binaryVars[i + (k + 1) * self.data.N],
                    name="zetaconstraint1_{" + str(i) + "," + str(k) + "}")
                self.zetaconstraint2[i + k * self.data.N] = self.mod.addConstr(
                    self.zetaVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.timeVars[i + k * self.data.N],
                    name="zetaconstraint2_{" + str(i) + "," + str(k) + "}")
                self.zetaconstraint3[i + k * self.data.N] = self.mod.addConstr(
                    self.zetaVars[i + k * self.data.N],
                    grb.GRB.GREATER_EQUAL,
                    self.timeVars[i + k * self.data.N] - (1 - self.binaryVars[i + (k + 1) * self.data.N]) * self.data.fractionUB,
                    name="zetaconstraint3_{" + str(i) + "," + str(k) + "}")
        self.zeeconstraints = [None] * (self.data.smallvoxelspace)
        ## Create a unique list of voxels (smallvoxelspace steps but living in bigvoxelspace)
        # This vector should be used later and is the standard ordering of particles in smallvoxelspace.
        voxels = np.unique(self.data.voxels)
        i = 0
        # Create all the dose constraints
        for voxel in voxels:
            # Find locations with value corresponding to voxel
            positions = np.where(voxel == self.data.voxels)[0]
            expr = grb.QuadExpr()
            for p in positions:
                abixel = self.data.bixels[p]
                expr += self.Dijs[abixel] * self.xiVars[abixel] * self.yVar
                expr += self.Dijs[abixel] * self.zetaVars[abixel] * self.yVar
            self.zeeconstraints[i] = self.mod.addConstr(self.zeeVars[i], grb.GRB.EQUAL, expr)
            i += 1
        # Make a lazy update of this last set of constraints
        self.mod.update()
        # Update the objective function.
        # Create a variable that will be the minimum dose to a PTV.
        self.minDosePTVVar = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, obj=0.0, vtype=grb.GRB.CONTINUOUS,
                                                                     name="minDosePTV", column=None)
        self.minDoseConstraints = []
        self.maxDoseConstraints = []
        for i in range(0, self.data.smallvoxelspace):
            # Constraint on minimum radiation if this is a tumor
            if 256 == self.data.mask[self.data.SmalltoBig[i]]:
                self.minDoseConstraints.append(self.mod.addConstr(self.minDosePTVVar, grb.GRB.LESS_EQUAL, self.zeeVars[i]))
            # Constraint on maximum radiation to the OAR
            if 2 == self.data.mask[self.data.SmalltoBig[i]]:
                self.maxDoseConstraints.append(self.mod.addConstr(self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.OARMAX))

        self.mod.update()

    def plotDVH(self, saveNameTag='', plotFull=False, showPlot=False, differentVoxelMap=''):
        if differentVoxelMap != '':
            voxMap = np.loadtxt(self.data.dataDirectory + differentVoxelMap, dtype='int32')
            voxDict = {}
            for t in self.data.targetIndices:
                voxDict[t] = np.where(voxMap == t)[0]
            for o in self.data.oarIndices:
                voxDict[o] = np.where(voxMap == o)[0]
        else:
            voxMap = np.loadtxt(self.data.dataDirectory + self.data.voxelMapDVH, dtype='int32')
            voxDict = {}
            for t in self.data.targetIndices:
                voxDict[t] = np.where(voxMap == t)[0]
            for o in self.data.oarIndices:
                voxDict[o] = np.where(voxMap == o)[0]
        dose = np.array([self.doseVars[j].X for j in range(self.data.nVox)])
        plt.clf()
        for index, sValues in voxDict.iteritems():
            if plotFull:
                sVoxels = sValues
            else:
                if not len(sValues) > 0:
                    continue
                sVoxels = sValues
            hist, bins = np.histogram(dose[sVoxels], bins=100)
            dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
            plt.plot(bins[:-1], dvh, label="struct " + str(index), linewidth=2)
        lgd = plt.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title('DVH')
        plt.savefig(self.data.outputDirectory + saveNameTag + '_' + self.data.outputFilename[:-4] + '.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
        if showPlot:
            plt.show()

    def outputSolution(self):
        outDict = {}
        outDict['yVector'] = np.array([self.cpBinaryVars[i].X for i in range(self.data.nCP)])
        outDict['doseVector'] = np.array([self.doseVars[j].X for j in range(self.data.nVox)])
        outDict['obj'] = self.mod.objVal

        if self.data.modType == 'modulated':
            outDict['intensities'] = np.array([self.cpBilinearVars[i].X for i in range(self.data.nCP)])

        # check if directory exists
        if not os.path.exists(self.data.outputDirectory):
            os.makedirs(self.data.outputDirectory)
        sio.savemat(self.data.outputDirectory + self.data.outputFilename, outDict)

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

class tomodata:
    ## Initialization of the data
    def __init__(self):
        ## M value. A large value that variable t will not reach (t in this case is from 0 to 1)
        self.fractionUB = 1.0
        self.N = 80
        ## Number of control points (every 2 degrees)
        self.K = 178
        ## Total number of beamlets
        self.totalbeamlets = self.K * self.N
        ## Dimensions of the 2-Dimensional screen
        self.dimX = 256
        self.dimY = 256
        ## Total number of voxels in the phantom
        self.totalVoxels = self.dimX * self.dimY
        self.bixels = getvector('data\\Bixels_out.bin', np.int32)
        self.voxels = getvector('data\\Voxels_out.bin', np.int32)
        self.Dijs = getvector('data\\Dijs_out.bin', np.float32)
        self.mask = getvector('data\\optmask.img', np.int32)
        ## This is the total number of voxels that there are in the body. Not all voxels from all directions.
        self.smallvoxelspace = len(np.unique(self.voxels))
        self.SmallToBigCreator()

    ## Create a map from big to small voxel space
    def SmallToBigCreator(self):
        self.SmalltoBig = np.where(0 != self.mask)

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
dataobject = tomodata()
print('done')
D = sparseWrapper(bixels, voxels, Dijs, mask, totalbeamlets, totalVoxels)

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(cwd, files)
