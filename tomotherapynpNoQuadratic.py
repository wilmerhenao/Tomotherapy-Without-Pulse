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
import pickle
from collections import defaultdict
import time
from multiprocessing import Pool
from functools import partial

numcores = 4
## Class definition of the gurobi object that handles creation and execution of the model
# Original template from Troy Long.
class tomotherapyNP(object):
    def __init__(self, datastructure):
        print('Reading in data...', end="")
        self.data = datastructure
        print('done')
        print('Constructing Gurobi model object...', end="")
        self.mod = grb.Model()
        self.mod.params.threads = numcores
        self.mod.params.MIPFocus = 3
        self.mod.params.PreSparsify = 1
        self.mod.params.Presolve = 2
        self.mod.params.TimeLimit = 4.0 # Time limit in seconds
        print('done')
        print('Building main decision variables (dose, binaries).')
        self.minDoseConstraints = []
        self.maxDoseConstraints = []
        self.buildVariables()
        self.launchOptimization()
        print('The problem has been completed')

    def addVarsandDoseConstraint(self):
        print('creating primary dose constraints...', end="")
        sys.stdout.flush()
        self.zeeconstraints = [None] * (self.data.totalsmallvoxels)
        ## This is the variable that will appear in the $z_{j}$ constraint. One per actual voxel in small space.
        self.zeeVars = [None] * (self.data.totalsmallvoxels)
        for i in range(0, self.data.totalsmallvoxels):
            self.zeeVars[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                                                     name="zee_{" + str(i) + "}",
                                                                     column = None)
        self.mod.update()
        for i in range(0, self.data.totalsmallvoxels):
            if i % 1000 == 0:
                print(str(i) + ',', end=""); sys.stdout.flush()
            self.zeeconstraints[i] = self.mod.addConstr(-self.zeeVars[i], grb.GRB.EQUAL, 0)
        # Lazy update of gurobi
        self.mod.update()
        # Addition of t variables. All terms according to the terminology in the writeup
        ## Binary Variable. I call it delta in the writeup
        self.binaryVars = [None] * (self.data.N * self.data.K)
        ## xi Variables. Helper variables to create a continuous binary variable
        self.xiVars = [None] * (self.data.N * self.data.K)
        ## IntensityVariable
        self.yVar = [None] * (self.data.K)
        ## mu Variables. Helper variables to remove the absolute value nonlinear constraint
        self.muVars = [None] * ((self.data.N) * (self.data.K - 1))
        for k in range(0, (self.data.K)):
            print('On control point: ' + str(k + 1) + ' out of ' + str(self.data.K))
            # yVar created outside the inner loop because there is only one per control point
            self.yVar[k] = self.mod.addVar(lb=0.0, ub=self.data.maxIntensity,
                                           vtype=grb.GRB.CONTINUOUS, name="Y_{" + str(k) + "}",
                                           column=None)
            # Create a partial function
            self.partialdeclareFunc = partial(self.declareVariables, k=k)
            for i in range(0, self.data.N):
                self.partialdeclareFunc(i)
                # if __name__ == '__main__':
                #      pool = Pool(processes=numcores)  # process per MP
                #      pool.map(self.partialdeclareFunc, range(0, self.data.N))
                # pool.close()
                # pool.join()
        # Lazy update of gurobi
        self.mod.update()

    def declareVariables(self, i, k):
        self.xiVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=self.data.maxIntensity,
                                                           vtype=grb.GRB.CONTINUOUS,
                                                           name="xi_{" + str(i) + "," + str(k) + "}",
                                                           column=grb.Column(np.array(self.data.D[:,
                                                                                      (i + k * self.data.N)].
                                                                                      todense().transpose())[
                                                                                 0].tolist(),
                                                                             self.zeeconstraints))

        self.binaryVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=1.0, vtype=grb.GRB.BINARY,
                                                               name="binary_{" + str(i) + "," + str(k) + "}",
                                                               column=None)
        # The mu variable will register change in the behaviour from one control to the other. Therefore loses 1
        # degree of freedom
        if (self.data.K - 1) != k:
            self.muVars[i + k * self.data.N] = self.mod.addVar(lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS,
                                                               name="mu_{" + str(i) + "," + str(k) + "}",
                                                               column=None)

    def createXiandAbsolute(self):
        # Add some constraints. This one is about replacing the absolute value with linear expressions
        self.absoluteValueRemovalConstraint1 = [None] * (self.data.N * (self.data.K-1))
        self.absoluteValueRemovalConstraint2 = [None] * (self.data.N * (self.data.K-1))
        # This constraint is about replacing the multiplication of binary times continuous variables using McCormick's envelopes
        self.xiconstraint1 = [None] * (self.data.N * self.data.K)
        self.xiconstraint2 = [None] * (self.data.N * self.data.K)
        self.xiconstraint3 = [None] * (self.data.N * self.data.K)
        # Constraints related to absolute value removal from objective function
        for k in range(0, (self.data.K - 1)):
            for i in range(0, self.data.N):
                # \mu \geq \beta_{i,k+1} - \beta_{i,k}
                self.absoluteValueRemovalConstraint1[i + k * self.data.N] = self.mod.addConstr(
                    self.muVars[i + k * self.data.N],
                    grb.GRB.GREATER_EQUAL,
                    self.binaryVars[i + (k + 1) * self.data.N] - self.binaryVars[i + k * self.data.N],
                    name = "rmabs1_{" + str(i) + "," + str(k) + "}")
                # \mu \geq -(\beta_{i,k+1} - \beta_{i,k})
                self.absoluteValueRemovalConstraint2[i + k * self.data.N] = self.mod.addConstr(
                    self.muVars[i + k * self.data.N],
                    grb.GRB.GREATER_EQUAL,
                    -(self.binaryVars[i + (k + 1) * self.data.N] - self.binaryVars[i + k * self.data.N]),
                    name = "rmabs2_{" + str(i) + "," + str(k) + "}")
        # Constraints related to McCormick relaxations.
        for k in range(0, self.data.K):
            for i in range(0, self.data.N):
                # \xi \leq \beta
                self.xiconstraint1[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.binaryVars[i + k * self.data.N],
                    name="xiconstraint1_{" + str(i) + "," + str(k) + "}")
                # \xi \leq y
                self.xiconstraint2[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.yVar[k],
                    name="xiconstraint2_{" + str(i) + "," + str(k) + "}")
                # \xi \geq y - (1 - \beta) U
                self.xiconstraint3[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.GREATER_EQUAL,
                    self.yVar[k] - (1 - self.binaryVars[i + k * self.data.N]) * self.data.maxIntensity,
                    name="xiconstraint3_{" + str(i) + "," + str(k) + "}")
        self.mod.update()

    def objConstraints(self):
        self.minDosePTVVar = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="minDosePTV", column=None)

        self.mod.update()
        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on minimum radiation if this is a tumor
            if self.data.mask[self.data.TumorMap[0][i]] in self.data.TARGETList:
                self.minDoseConstraints.append(self.mod.addConstr(self.minDosePTVVar, grb.GRB.LESS_EQUAL, self.zeeVars[i]))
            # Constraint on maximum radiation to the OAR
            if self.data.mask[self.data.TumorMap[0][i]] in self.data.OARList:
                self.maxDoseConstraints.append(self.mod.addConstr(self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.OARMAX))
        self.mod.update()

    ## This function builds variables to be included in the model
    def buildVariables(self):
        print('\nBuilding Variables related to dose constraints...')
        sys.stdout.flush()
        self.addVarsandDoseConstraint()
        print('done')
        print('Building Secondary constraints; binaries, mu, xi...', end="")
        self.createXiandAbsolute()
        print('done')
        # Update the objective function.
        # Create a variable that will be the minimum dose to a PTV.
        print('creating VOI constraints and constraints directly associated with the objective...', end="")
        self.objConstraints()
        print('done')
        # Set the objective value

    def launchOptimization(self):
        print('Setting up and launching the optimization...', end="")
        objexpr = grb.LinExpr()
        objexpr = -self.minDosePTVVar

        self.mod.setObjective(objexpr, 1.0) #1.0 expresses minimization. It is the model sense.
        self.mod.update()
        self.mod.optimize()
        print('done')

    def outputSolution(self):
        outDict = {}
        outDict['yVector'] = np.array([self.cpBinaryVars[i].X for i in range(self.data.K)])
        outDict['doseVector'] = np.array([self.zeeVars[j].X for j in range(self.data.totalsmallvoxels)])
        outDict['obj'] = self.mod.objVal

        if self.data.modType == 'modulated':
            outDict['intensities'] = np.array([self.cpBilinearVars[i].X for i in range(self.data.nCP)])

        # check if directory exists
        if not os.path.exists(self.data.outputDirectory):
            os.makedirs(self.data.outputDirectory)
        sio.savemat(self.data.outputDirectory + self.data.outputFilename, outDict)

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
        outDict['doseVector'] = np.array([self.doseVars[j].X for j in range(self.data.nVox)])
        outDict['obj'] = self.mod.objVal

## Function that reads the files produced by Weiguo
def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)

## Function that takes a list and returns a default dictionary with repeated indices as duplicates
def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key,locs in tally.items()
                            if len(locs) > 1)

class tomodata:
    ## Initialization of the data
    def __init__(self):
        ## C Value in the objective function
        self.C = 1.0
        ## M value. A large value that variable t will not reach (t in this case is from 0 to 1)
        self.fractionUB = 1.0
        self.maxIntensity = 10.0
        self.N = 80
        ## Number of control points (every 2 degrees)
        self.K = 178
        ## Total number of beamlets
        ## OARMAX is maximum dose tolerable for organs. 10 in this case
        self.OARMAX = 10
        print('Read vectors...', end="")
        self.readWilmersCase()
        #self.readWeiguosCase()
        sys.stdout.flush()
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        print('Build sparse matrix.')
        # The next part uses the case corresponding to either Wilmer or Weiguo's case.
        self.totalbeamlets = self.K * self.N
        self.totalsmallvoxels = max(self.smallvoxels) + 1
        print('lengts of voxels and mask: ' + str(len((self.smallvoxels))) + ' ' + str(len(self.mask)))
        print("unique voi's: ", np.unique(self.mask))
        self.D = sps.csc_matrix((self.Dijs, (self.smallvoxels, self.bixels)), shape=(self.totalsmallvoxels, self.totalbeamlets))
        ## This is the total number of voxels that there are in the body. Not all voxels from all directions
        self.TumorMapCreator()

    ## Create a map from big to small voxel space
    def TumorMapCreator(self):
        self.TumorMap = np.where(0 != self.mask)

    ## Create a map from big to small voxel space
    def BigToSmallCreator(self):
        a,b,c,d = np.unique(self.voxels, return_index=True, return_inverse=True, return_counts=True)
        return(c)

    def readWeiguosCase(self):
        self.bixels = getvector('data\\Bixels_out.bin', np.int32)
        self.voxels = getvector('data\\Voxels_out.bin', np.int32)
        self.Dijs = getvector('data\\Dijs_out.bin', np.float32)
        self.mask = getvector('data\\optmask.img', np.int32)
        self.OARList = [2, 3]
        self.TARGETList = [256]

    def readWilmersCase(self):
        self.bixels = getvector('data\\myBixels_out.bin', np.int32)
        self.voxels = getvector('data\\myVoxels_out.bin', np.int32)
        self.Dijs = getvector('data\\myDijs_out.bin', np.float32)
        self.mask = getvector('data\\myoptmask.img', np.int32)
        with open('C:/Users/S170452/PycharmProjects/Tomotherapy-Without-Pulse/data/mydict.pckl', 'rb') as ff:
            dictdata = pickle.load(ff)
        self.K = dictdata['K']
        self.N = dictdata['N']
        self.OARList = dictdata['OARIDs']
        self.TARGETList = dictdata['TARGETIDs']
        print("Targets and OARS: ", self.TARGETList, self.OARList)

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
start_time = time.time()
dataobject = tomodata()
tomoinstance = tomotherapyNP(dataobject)
print("--- %s seconds ---" % (time.time() - start_time))