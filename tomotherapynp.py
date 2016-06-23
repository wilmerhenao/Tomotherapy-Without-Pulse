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
        voxels = np.unique(self.data.voxels)
        i = 0
        # Create all the dose constraints
        for voxel in voxels:
            # Find locations with value corresponding to voxel
            positions = np.where(voxel == self.data.voxels)[0]
            expr = grb.LinExpr()
            for p in positions:
                abixel = self.data.bixels[p]
                expr += self.Dijs[abixel] * self.xiVars[abixel] * self.yVar
                expr += self.Dijs[abixel] * self.zetaVars[abixel] * self.yVar
            self.zeeconstraints[i] = self.mod.addConstr(self.zeeVars[i], grb.GRB.LESS_EQUAL, expr )
            i += 1
        # Make a lazy update of this last set of constraints
        self.mod.update()



        self.cpBinaryVars = [None] * self.data.nCP
        print 'Building dose constraint (of ', self.data.nCP, 'for CP',
        for i in range(self.data.nCP):
            if i % 500 == 0:
                print str(i), ',',
            self.cpBinaryVars[i] = self.mod.addVar(vtype=grb.GRB.BINARY, column=grb.Column(
                np.array(self.data.D[:, i].todense().transpose())[0].tolist(), self.doseConstraint))
            # self.cpBinaryVars[i] = self.mod.addVar(vtype=grb.GRB.BINARY, name='y' + str(i))
        print 'done'
        self.mod.update()

    def buildVariablesModulated(self):
        self.doseVars = [self.mod.addVar() for j in range(self.data.nVox)]
        self.mod.update()
        self.doseConstraint = [self.mod.addConstr(-self.doseVars[j], grb.GRB.EQUAL, 0) for j in range(self.data.nVox)]
        self.mod.update()
        self.cpBinaryVars = [None] * self.data.nCP
        self.cpBilinearVars = [None] * self.data.nCP
        self.cpIntensityVars = [None] * self.data.nCP
        print 'Building dose constraint (of ', self.data.nCP, 'for CP',
        for i in range(self.data.nCP):
            if i % 500 == 0:
                print str(i), ',',
            self.cpBinaryVars[i] = self.mod.addVar(vtype=grb.GRB.BINARY)
            self.cpIntensityVars[i] = self.mod.addVar(vtype=grb.GRB.CONTINUOUS, ub=self.data.intensityUB,
                                                      lb=self.data.intensityLB)
            self.cpBilinearVars[i] = self.mod.addVar(vtype=grb.GRB.CONTINUOUS, column=grb.Column(
                np.array(self.data.D[:, i].todense().transpose())[0].tolist(), self.doseConstraint))
            # self.cpBinaryVars[i] = self.mod.addVar(vtype=grb.GRB.BINARY, name='y' + str(i))
        print 'done'
        self.mod.update()
        print 'Building bilinear constraints'
        for i in range(self.data.nCP):
            self.mod.addConstr(self.cpBilinearVars[i], grb.GRB.LESS_EQUAL, self.cpIntensityVars[i])
            self.mod.addConstr(self.cpBilinearVars[i], grb.GRB.LESS_EQUAL, self.cpBinaryVars[i] * self.data.intensityUB)
            self.mod.addConstr(self.cpBilinearVars[i], grb.GRB.GREATER_EQUAL,
                               self.cpIntensityVars[i] - self.data.intensityUB * (1 - self.cpBinaryVars[i]))
        self.mod.update()

    def buildPath(self, pathType='gIcU'):
        print 'Building path type', pathType, '...',
        if pathType == 'gIcU':
            # build one-CP-per-couch-angle constraint
            self.gIcU_singleGanryPerCouchAngle = [self.mod.addConstr(grb.quicksum([self.cpBinaryVars[i] for i in range(
                self.data.numCouchPerGantryCum[g], self.data.numCouchPerGantryCum[g + 1])]), grb.GRB.EQUAL, 1) for g in
                                                  range(self.data.nGantry)]
            # build feasible path constraint

            self.gIcU_feasConstraint = [None] * np.sum(self.data.numCouchPerGantry[:-1])
            for g in range(self.data.nGantry - 1):
                for counter in range(self.data.numCouchPerGantryCum[g], self.data.numCouchPerGantryCum[g + 1]):
                    # counter = self.data.numCouchPerGantryCum[g] + c
                    self.gIcU_feasConstraint[counter] = self.mod.addConstr(self.cpBinaryVars[counter] + grb.quicksum(
                        [self.data.adjacency[counter, i] * self.cpBinaryVars[i] for i in
                         range(self.data.numCouchPerGantryCum[g + 1], self.data.numCouchPerGantryCum[g + 2]) if
                         self.data.adjacency[counter, i] > 0]), grb.GRB.LESS_EQUAL, 1)

            self.mod.update()

        if pathType == 'gUcI':
            '''
            Note that numCouchPerGantry is really numGantryPerCouch
            '''
            # build one-CP-per-couch-angle constraint
            self.gUcI_singleCouchPerGantryAngle = [self.mod.addConstr(grb.quicksum([self.cpBinaryVars[i] for i in range(
                self.data.numCouchPerGantryCum[g], self.data.numCouchPerGantryCum[g + 1])]), grb.GRB.EQUAL, 1) for g in
                                                   range(self.data.nGantry)]
            # build feasible path constraint

            self.gUcI_feasConstraint = [None] * np.sum(self.data.numCouchPerGantry[:-1])
            for g in range(self.data.nGantry - 1):
                for counter in range(self.data.numCouchPerGantryCum[g], self.data.numCouchPerGantryCum[g + 1]):
                    # counter = self.data.numCouchPerGantryCum[g] + c
                    self.gUcI_feasConstraint[counter] = self.mod.addConstr(self.cpBinaryVars[counter] + grb.quicksum(
                        [self.data.adjacency[counter, i] * self.cpBinaryVars[i] for i in
                         range(self.data.numCouchPerGantryCum[g + 1], self.data.numCouchPerGantryCum[g + 2]) if
                         self.data.adjacency[counter, i] > 0]), grb.GRB.LESS_EQUAL, 1)

            self.mod.update()

        print 'done'

    def buildProtocol(self):
        # for each line in the protocol, call a function to build each particular constraint/objective
        # keys = ["protID","structID","sType","metric","bound","type","weight"]
        self.protocolConstraints = {}
        self.protocolVars = {}
        for pNumber, pValues in self.data.protDict.iteritems():
            print pNumber, pValues

            # build hard constraint
            if pValues['type'] == 'HC':
                # build max/min constraint
                if pValues['metric'] == 'max':
                    print pNumber, 'building max bound for structure', pValues['structID']
                    self.buildMaxMinHardConstraint(pValues['protID'], pValues['structID'], pValues['bound'], type='max')
                if pValues['metric'] == 'min':
                    print pNumber, 'building min bound for structure', pValues['structID']
                    self.buildMaxMinHardConstraint(pValues['protID'], pValues['structID'], pValues['bound'], type='min')
                if pValues['metric'] == 'meanUB':
                    print pNumber, 'building mean upper bound for structure', pValues['structID']
                    self.buildMeanHardConstraint(pValues['protID'], pValues['structID'], pValues['bound'], type='max')
                if pValues['metric'] == 'meanLB':
                    print pNumber, 'building mean lower bound for structure', pValues['structID']
                    self.buildMeanHardConstraint(pValues['protID'], pValues['structID'], pValues['bound'], type='min')

            # build soft constraint

            if pValues['type'] == 'SC':
                # build max/min soft constraints
                if pValues['metric'] == 'max':
                    print pNumber, 'building soft max bound for structure', pValues['structID']
                    self.buildMaxMinSoftConstraint(pValues['protID'], pValues['structID'], pValues['bound'],
                                                   pValues['weight'], type='max')
                if pValues['metric'] == 'min':
                    print pNumber, 'building soft min bound for structure', pValues['structID']
                    self.buildMaxMinSoftConstraint(pValues['protID'], pValues['structID'], pValues['bound'],
                                                   pValues['weight'], type='min')
                if pValues['metric'] == 'meanUB':
                    print pNumber, 'building soft mean bound for structure', pValues['structID']
                    self.buildMeanSoftConstraint(pValues['protID'], pValues['structID'], pValues['bound'],
                                                 pValues['weight'], type='max')
                if pValues['metric'] == 'meanLB':
                    print pNumber, 'building soft mean bound for structure', pValues['structID']
                    self.buildMeanSoftConstraint(pValues['protID'], pValues['structID'], pValues['bound'],
                                                 pValues['weight'], type='min')

            print pNumber, pValues['type']

    def buildMaxMinHardConstraint(self, id, structure, bound, type='min'):
        vars = []
        const = []
        boundType = 'lb'
        if type == 'max':
            boundType = 'ub'

        if type=='min':
            const.append(self.mod.addConstr(grb.quicksum([self.doseVars[j] for j in self.data.structureIndexDict[structure]]),grb.GRB.GREATER_EQUAL,bound *len(self.data.structureIndexDict[structure])))
        elif type=='max':
            const.append(self.mod.addConstr(grb.quicksum([self.doseVars[j] for j in self.data.structureIndexDict[structure]]),grb.GRB.LESS_EQUAL,bound * len(self.data.structureIndexDict[structure])))
        self.mod.update()
        self.protocolVars[id] = vars
        self.protocolConstraints[id] = const

    def buildMaxMinHardConstraint(self, id, structure, bound, type='min'):
        vars = []
        const = []
        boundType = 'lb'
        if type == 'max':
            boundType = 'ub'
        for j in self.data.structureIndexDict[structure]:
            self.doseVars[j].setAttr(boundType, bound)
        self.mod.update()
        self.protocolVars[id] = vars
        self.protocolConstraints[id] = const

    def buildMaxMinSoftConstraint(self, id, structure, bound, penalty, type='min'):
        vars = []
        const = []
        boundType = 'lb'
        if type == 'max':
            boundType = 'ub'

        # generate penalty variable
        vars.append(self.mod.addVar(obj=penalty, name=str(id) + boundType))
        self.mod.update()

        if type == 'max':
            for j in self.data.structureIndexDict[structure]:
                const.append(self.mod.addConstr(self.doseVars[j], grb.GRB.LESS_EQUAL, bound + vars[0]))
        elif type == 'min':
            for j in self.data.structureIndexDict[structure]:
                const.append(self.mod.addConstr(self.doseVars[j], grb.GRB.GREATER_EQUAL, bound - vars[0]))
        self.mod.update()
        self.protocolVars[id] = vars
        self.protocolConstraints[id] = const

    def buildMeanSoftConstraint(self, id, structure, bound, penalty, type='min'):
        vars = []
        const = []
        boundType = 'meanlb'
        if type == 'max':
            boundType = 'meanub'

        # generate penalty variable

        vars.append(self.mod.addVar(obj=penalty, name=str(id) + boundType))
        vars.append(self.mod.addVar(name=str(id) + 'mean'))
        self.mod.update()

        # build mean
        const.append(self.mod.addConstr(len(self.data.structureIndexDict[structure]) * vars[1], grb.GRB.EQUAL,
                                        grb.quicksum(
                                            [self.doseVars[j] for j in self.data.structureIndexDict[structure]])))

        if type == 'max':
            const.append(self.mod.addConstr(vars[1], grb.GRB.LESS_EQUAL, bound + vars[0]))
        elif type == 'min':
            const.append(self.mod.addConstr(vars[1], grb.GRB.GREATER_EQUAL, bound - vars[0]))

        self.mod.update()
        self.protocolVars[id] = vars
        self.protocolConstraints[id] = const

    def buildObjective(self, objDict):
        objType = objDict['objType']
        print 'Building objective type', objType, '...',

        if objType == 'minTumorMaxOar':
            # set objective

            objExpr = grb.LinExpr(0.0)
            # for each target, gen min dose var, add constraints, add term to obj
            self.tMinConst = [[self.mod.addConstr(0, grb.GRB.LESS_EQUAL, self.doseVars[j]) for j in
                               self.data.targetStructureIndexDict[self.data.targetIndices[t]]] for t in
                              range(len(self.data.targetIndices))]

            self.mod.update()
            self.tMinVars = [
                self.mod.addVar(obj=-1.0 * objDict['t' + str(self.data.targetIndices[t])], column=grb.Column(
                    [1 for j in range(len(self.data.targetStructureIndexDict[self.data.targetIndices[t]]))],
                    self.tMinConst[t])) for t in range(len(self.data.targetIndices))]
            self.mod.update()

            self.oMaxConst = [[self.mod.addConstr(0, grb.GRB.GREATER_EQUAL, self.doseVars[j]) for j in
                               self.data.oarStructureIndexDict[self.data.oarIndices[t]]] for t in
                              range(len(self.data.oarIndices))]
            self.mod.update()
            self.oMaxVars = [self.mod.addVar(obj=1.0 * objDict['o' + str(self.data.oarIndices[t])], column=grb.Column(
                [1 for j in range(len(self.data.oarStructureIndexDict[self.data.oarIndices[t]]))], self.oMaxConst[t]))
                             for t in range(len(self.data.oarIndices))]
            self.mod.update()

        if objType == 'linEud':
            minPTVScalar = 0.8
            # for each target, gen min dose var, add constraints, add term to obj
            self.tMinConst = [[self.mod.addConstr(0, grb.GRB.LESS_EQUAL, self.doseVars[j]) for j in
                               self.data.targetStructureIndexDict[self.data.targetIndices[t]]] for t in
                              range(len(self.data.targetIndices))]

            self.mod.update()
            self.tMinVars = [self.mod.addVar(obj=-1.0 * minPTVScalar * objDict['t' + str(self.data.targetIndices[t])],
                                             column=grb.Column(
                                                 [1 for j in range(len(
                                                     self.data.targetStructureIndexDict[self.data.targetIndices[t]]))],
                                                 self.tMinConst[t])) for t in range(len(self.data.targetIndices))]
            self.mod.update()
            self.tMeanVarConst = [self.mod.addConstr(grb.quicksum(
                [self.doseVars[j] for j in range(len(self.data.targetStructureIndexDict[self.data.targetIndices[t]]))]),
                grb.GRB.EQUAL, 0) for t in range(len(self.data.targetIndices))]
            self.mod.update()
            self.tMeanVars = [
                self.mod.addVar(obj=-1.0 * (1. - minPTVScalar) * objDict['t' + str(self.data.targetIndices[t])],
                                column=grb.Column(
                                    -len(self.data.targetStructureIndexDict[self.data.targetIndices[t]]),
                                    self.tMeanVarConst[t])) for t in range(len(self.data.targetIndices))]
            self.mod.update()

            maxOARScalar = 0.3
            self.oMaxConst = [[self.mod.addConstr(0, grb.GRB.GREATER_EQUAL, self.doseVars[j]) for j in
                               self.data.oarStructureIndexDict[self.data.oarIndices[t]]] for t in
                              range(len(self.data.oarIndices))]

            self.mod.update()
            self.oMaxVars = [
                self.mod.addVar(obj=1.0 * maxOARScalar * objDict['o' + str(self.data.oarIndices[t])], column=grb.Column(
                    [1 for j in range(len(self.data.oarStructureIndexDict[self.data.oarIndices[t]]))],
                    self.oMaxConst[t]))
                for t in range(len(self.data.oarIndices))]
            self.mod.update()

            self.oMeanVarConst = [self.mod.addConstr(grb.quicksum(
                [self.doseVars[j] for j in range(len(self.data.oarStructureIndexDict[self.data.oarIndices[t]]))]),
                grb.GRB.EQUAL, 0) for t in range(len(self.data.oarIndices))]
            self.mod.update()
            self.oMeanVars = [
                self.mod.addVar(obj=1.0 * (1. - maxOARScalar) * objDict['o' + str(self.data.oarIndices[t])],
                                column=grb.Column(
                                    -len(self.data.oarStructureIndexDict[self.data.oarIndices[t]]),
                                    self.oMeanVarConst[t])) for t in range(len(self.data.oarIndices))]
            self.mod.update()

        if objType == 'feasCheck':
            self.mod.setObjective(0.0, grb.GRB.MINIMIZE)
            self.mod.update()

        print 'done'

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

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
dataobject = tomodata()

D = sparseWrapper(bixels, voxels, Dijs, mask, totalbeamlets, totalVoxels)

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(cwd, files)