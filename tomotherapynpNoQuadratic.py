#!/home/wilmer/anaconda3/bin/python

__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

#from data import *
import gurobipy as grb
from scipy.spatial import KDTree
import numpy.ma as ma
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
import math
from scipy.stats import describe
import matplotlib
import re

numcores = 4

def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        mipgap = 0.02 # Here 0.2 means 2%
        if 0 != objbst:
            if abs(objbst - objbnd)/abs(objbst) < mipgap:
                print('Stop early -', str(mipgap * 100), '% gap achieved')
                model.terminate()

## Class definition of the gurobi object that handles creation and execution of the model
# Original template from Troy Long.
class tomotherapyNP(object):
    def __init__(self, datastructure):
        print('Reading in data...')
        self.data = datastructure
        print('done')
        print('Constructing Gurobi model object...')
        self.mod = grb.Model()
        self.mod.params.threads = numcores
        self.mod.params.MIPFocus = 1
        print('done')
        print('Building main decision variables (dose, binaries).')
        self.buildVariables()
        self.launchOptimizationPWLOwnImplementation()
        self.plotDVH('dvhcheck')
        self.plotSinoGram()
        self.plotEventsbinary()
        print('The problem has been completed')

    def addVarsandDoseConstraint(self):
        print('creating primary dose constraints...')
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
                print(str(i) + ','); sys.stdout.flush()
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
            print('Loading control point: ' + str(k + 1) + ' out of ' + str(self.data.K))
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

        self.binaryVars[i + k * self.data.N] = self.mod.addVar(vtype=grb.GRB.BINARY,
                                                               name="binary_{" + str(i) + "," + str(k) + "}",
                                                               column=None)
        # The mu variable will register change in the behaviour from one control to the other. Therefore loses 1
        # degree of freedom
        if (self.data.K - 1) != k:
            self.muVars[i + k * self.data.N] = self.mod.addVar(vtype=grb.GRB.BINARY,
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
        self.sumMaxRestriction = [None] * (self.data.N)
        # Constraints related to absolute value removal from objective function
        for i in range(0, self.data.N):
            expr = grb.LinExpr()
            for k in range(0, (self.data.K - 1)):
                # sum mu variables and restrict their sum to be smaller than M
                expr += self.muVars[i + k * self.data.N]
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
            self.sumMaxRestriction[i] = self.mod.addConstr(expr, grb.GRB.LESS_EQUAL,self.data.M,
                                                                   name = "summaxrest_{" + str(i) + "}")
            expr = None

        # Constraints related to McCormick relaxations.
        for k in range(0, self.data.K):
            for i in range(0, self.data.N):
                # \xi \leq \beta
                self.xiconstraint1[i + k * self.data.N] = self.mod.addConstr(
                    self.xiVars[i + k * self.data.N],
                    grb.GRB.LESS_EQUAL,
                    self.binaryVars[i + k * self.data.N] * self.data.maxIntensity,
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

    def objConstraintsMinDose(self):
        self.minDoseConstraints = []
        self.maxDoseConstraints = []
        self.minDosePTVVar = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="minDosePTV", column = None)
        self.objQuad = grb.QuadExpr()
        self.mod.update()
        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on minimum radiation if this is a tumor
            if self.data.mask[i] in self.data.TARGETList:
                self.minDoseConstraints.append(self.mod.addConstr(self.minDosePTVVar, grb.GRB.LESS_EQUAL,
                                                                  self.zeeVars[i]))
                # Constraint even the maximum dose to voxels
                self.minDoseConstraints.append(self.mod.addConstr(
                    self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.maxDosePTV))
            # Constraint on maximum radiation to the OAR
            elif self.data.mask[i] in self.data.OARList:
                self.maxDoseConstraints.append(self.mod.addConstr(self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.OARMAX))
            elif 0 == self.data.mask[i] :
                print('there is an element in the voxels that is also mask 0')
                ## sys.exit('there is a voxel that does not belong anywhere')
        #self.mod.addConstr(self.minDosePTVVar, grb.GRB.GREATER_EQUAL, 8.00)
        self.mod.update()
    # def objConstraintsQuadratic(self):
    #     self.QuadDosePTVVar = self.mod.addVar(lb = 0.0, ub = grb.GRB.INFINITY, vtype = grb.GRB.CONTINUOUS,
    #                                          name = "minDosePTV", column = None)
    #     self.mod.update()
    #     for i in range(0, self.data.totalsmallvoxels):
    #         # Constraint on minimum radiation if this is a tumor
    #         if self.data.mask[i] in self.data.TARGETList:
    #             self.minDoseConstraints.append(self.mod.addConstr(self.minDosePTVVar, grb.GRB.LESS_EQUAL,
    #                                                               self.zeeVars[i] ) )
    #             # Constraint even the maximum dose to voxels
    #             self.minDoseConstraints.append(self.mod.addConstr(
    #                 self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.maxDosePTV))
    #         # Constraint on maximum radiation to the OAR
    #         elif self.data.mask[i] in self.data.OARList:
    #             self.maxDoseConstraints.append(
    #                 self.mod.addConstr(self.zeeVars[i], grb.GRB.LESS_EQUAL, self.data.OARMAX
    #                                    )
    #                 )
    #         else:
    #             sys.exit('there is a voxel that does not belong anywhere')
    #     # self.mod.addConstr(self.minDosePTVVar, grb.GRB.GREATER_EQUAL, 8.00)
    #     self.mod.update()

    ## This function builds variables to be included in the model
    def buildVariables(self):
        print('\nBuilding Variables related to dose constraints...')
        sys.stdout.flush()
        self.addVarsandDoseConstraint()
        print('done')
        print('Building Secondary constraints; binaries, mu, xi...')
        self.createXiandAbsolute()
        print('done')
        # Update the objective function.
        # Create a variable that will be the minimum dose to a PTV.
        # Set the objective value

    def objConstraintsPWLOwnImplementation(self):
        # Create the auxiliary variables first in order to minimize the calls to update
        self.y1 = [None] * self.data.totalsmallvoxels
        self.y2 = [None] * self.data.totalsmallvoxels
        self.wbinary1 = [None] * self.data.totalsmallvoxels
        # Variable definition
        for i in range(0, self.data.totalsmallvoxels):
            if self.data.mask[i] in self.data.TARGETList:
                T = self.data.TARGETThresholds[np.where(self.data.mask[i] == self.data.TARGETList)[0][0]]
                self.y1[i] = self.mod.addVar(lb=0.0, ub=T, vtype=grb.GRB.CONTINUOUS,
                                             name="Auxiliaryy1Voxel_{" + str(i) + "}", column=None)
                self.y2[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="Auxiliaryy2Voxel_{" + str(i) + "}", column=None)
                self.wbinary1[i] = self.mod.addVar(vtype=grb.GRB.BINARY, name="wbinaryauxiliar_{" + str(i) + "}", column=None)
            elif self.data.mask[i] in self.data.OARList:
                T = self.data.OARThresholds[np.where(self.data.mask[i] == self.data.OARList)[0][0]]
                self.y1[i] = self.mod.addVar(lb=0.0, ub=T, vtype=grb.GRB.CONTINUOUS,
                                             name="Auxiliaryy1Voxel_{" + str(i) + "}", column=None)
                self.y2[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="Auxiliaryy2Voxel_{" + str(i) + "}", column=None)
                self.wbinary1[i] = self.mod.addVar(vtype=grb.GRB.BINARY, name="wbinaryauxiliar_{" + str(i) + "}", column=None)
        self.mod.update()

        # Constraints implementation
        self.y1constraint = [None] * self.data.totalsmallvoxels
        self.y2constraint1 = [None] * self.data.totalsmallvoxels
        self.y2constraint2 = [None] * self.data.totalsmallvoxels
        self.sumconstraint = [None] * self.data.totalsmallvoxels
        self.interceptions = [None] * self.data.totalsmallvoxels
        objexpr = grb.LinExpr()

        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on TARGETS
            if self.data.mask[i] in self.data.TARGETList:
                T = self.data.TARGETThresholds[np.where(self.data.mask[i] == self.data.TARGETList)[0][0]]
                self.y1constraint[i] = self.mod.addConstr(T * self.wbinary1[i], grb.GRB.LESS_EQUAL, self.y1[i])
                self.y2constraint1[i] = self.mod.addConstr(self.y2[i], grb.GRB.LESS_EQUAL, 2.0 * T * self.wbinary1[i])
                self.y2constraint2[i] = self.mod.addConstr(0.0, grb.GRB.LESS_EQUAL, 2.0 * T * self.wbinary1[i])
                self.sumconstraint[i] = self.mod.addConstr(self.zeeVars[i], grb.GRB.EQUAL, self.y1[i] + self.y2[i])
                objexpr += 1000.0 * T - 1000.0 * self.y1[i] + self.y2[i]
            # Constraint on OARs
            elif self.data.mask[i] in self.data.OARList:
                T = self.data.OARThresholds[np.where(self.data.mask[i] == self.data.OARList)[0][0]]
                self.y1constraint[i] = self.mod.addConstr(T * self.wbinary1[i], grb.GRB.LESS_EQUAL, self.y1[i])
                self.y2constraint1[i] = self.mod.addConstr(self.y2[i], grb.GRB.LESS_EQUAL, 2.0 * T * self.wbinary1[i])
                self.y2constraint2[i] = self.mod.addConstr(0.0, grb.GRB.LESS_EQUAL, 2.0 * T * self.wbinary1[i])
                self.sumconstraint[i] = self.mod.addConstr(self.zeeVars[i], grb.GRB.EQUAL, self.y1[i] + self.y2[i])
                objexpr += T - 1.0 * self.y1[i] + 1000.0 * self.y2[i]
            elif 0 == self.data.mask[i]:
                print('there is an element in the voxels that is also mask 0')

        self.mod.setObjective(objexpr, grb.GRB.MINIMIZE)  # 1.0 expresses minimization. It is the model sense.
        self.mod.update()

    def compareHDandSmallSpace(self, stepcoarse, stephd):
        om = [ij for ij in range(self.data.voxelsBigSpace)]
        print(om)
        # New Map
        nm = []
        # indicescoarse contains address in bigvoxelspace of the coarse grid
        # indiceshd contains address in bigvoxelspace of the hd grid
        for i in np.arange(0, self.data.caseSide, stepcoarse):
            for j in np.arange(0, self.data.caseSide, stepcoarse):
                nm.append(om[int(j) + int(i) * self.data.caseSide])
        # indices ONLY contains those indices determined by the step size.
        indices = np.where(np.in1d(self.data.voxelsHD, nm))[0]
        # indices coarse represents those represented by step size AND which have important data to show.
        self.indicescoarse = np.unique(self.data.voxelsHD[indices])

        print(indices, self.indicescoarse)
        print('length indicescoarse: ', len(self.indicescoarse) )
        nm = []
        for i in np.arange(0, self.data.caseSide, stephd):
            for j in np.arange(0, self.data.caseSide, stephd):
                nm.append(om[int(j) + int(i) * self.data.caseSide])
        indices = np.where(np.in1d(self.data.voxelsHD, nm))[0]
        self.indicesfine = np.unique(self.data.voxelsHD[indices])
        print('length indiceshd: ', len(self.indicesfine) )
        self.hd = []
        for i in self.indicesfine:
            if 0 == len(np.where(i == self.indicescoarse)[0]):
                self.hd.append(None)
            else:
                self.hd.append(np.where(i == self.indicescoarse)[0][0])
        self.hd = np.array(self.hd)

    def loadprevioussolution(self):
        if os.path.isfile("solutionStep-" + str(self.data.coarse) + ".sol"):
            self.compareHDandSmallSpace(self.data.coarse, self.data.sampleevery)
            # Load the previous file, Some values will have to be overwritten later.
            self.mod.read("solutionStep-" + str(self.data.coarse) + ".sol")
            # Now rewrite the data.
            print('reWrite invalid data')
            for i in range(self.data.totalsmallvoxels):
                self.zeeVars[i].Start = grb.GRB.UNDEFINED
                self.y1[i].Start = grb.GRB.UNDEFINED
                self.y2[i].Start = grb.GRB.UNDEFINED
                self.wbinary1[i].Start = grb.GRB.UNDEFINED
            print('done rewriting invalid data')
            # Give these branches the highest priority. Doesn't seem to have a big impact.
            self.mod.setAttr("BranchPriority", self.zeeVars, [2] * self.data.totalsmallvoxels )
            self.mod.setAttr("BranchPriority", self.y1, [1] * self.data.totalsmallvoxels )
            self.mod.setAttr("BranchPriority", self.y2, [1] * self.data.totalsmallvoxels )
            file = open("solutionStep-" + str(self.data.coarse) + ".sol", 'r' )
            #This container will preserve values to be used as hints
            zeehintmaker = []
            for linelong in file:
                linelong = linelong.split()
                line = linelong[0]
                token = line[:3]
                if 'zee' == token:
                    # Get the old position of the line (integer value inside the brackets from the sol file
                    posold = int(re.sub(r'[\{\}]', ' ', line).split()[1])
                    # Find where in hd this positiion belongs to.
                    posnew = np.where(self.hd == posold)[0][0]
                    # Grab character between brackets
                    self.zeeVars[posnew].Start = float(linelong[1])
                    #self.mod.setAttr("Start", self.zeeVars[posnew], [float(linelong[1])])
                    self.zeeVars[posnew].setAttr("Start", float(linelong[1]))
                    self.zeeVars[posnew].setAttr("VarHintVal", float(linelong[1]))
                    self.zeeVars[posnew].setAttr("VarHintPri", 10)
                    zeehintmaker.append(float(linelong[1]))
                    # Reduce this branch to a normal priority
                    self.zeeVars[posnew].BranchPriority = 0
                    print('zee replaced', linelong)
                elif 'wbi' == token:
                    # Get the old position of the line (integer value inside the brackets from the sol file
                    posold = int(re.sub(r'[\{\}]', ' ', line).split()[1])
                    # Find where in hd this positiion belongs to.
                    posnew = np.where(self.hd == posold)[0][0]
                    # Grab character between brackets
                    self.wbinary1[posnew].Start = float(linelong[1])
                    print('is the binary assignation happening?', self.wbinary1[posnew].getAttr("Start"), float(linelong[1]))
                    # Reduce this branch to a normal priority
                    print('wbi replaced', linelong)
                elif 'Aux' == token:
                    token = line[10]
                    if '1' == token:
                        # Get the old position of the line (integer value inside the brackets from the sol file
                        posold = int(re.sub(r'[\{\}]', ' ', line).split()[1])
                        # Find where in hd this positiion belongs to.
                        posnew = np.where(self.hd == posold)[0][0]
                        # Grab character between brackets
                        self.y1[posnew].Start = float(linelong[1])
                        # Reduce this branch to a normal priority
                        self.y1[posnew].BranchPriority = 0
                        # Reduce this branch to a normal priority
                        print('Aux1 replaced', linelong)
                    elif '2' == token:
                        # Get the old position of the line (integer value inside the brackets from the sol file
                        posold = int(re.sub(r'[\{\}]', ' ', line).split()[1])
                        # Find where in hd this positiion belongs to.
                        posnew = np.where(self.hd == posold)[0][0]
                        # Grab character between brackets
                        self.y2[posnew].Start = float(linelong[1])
                        # Reduce this branch to a normal priority
                        self.y2[posnew].BranchPriority = 0
                        # Reduce this branch to a normal priority
                        print('Aux2 replaced', linelong)
                    else:
                        print('error. This variable does not make sense', token, line)
                else:
                    pass
            print('Filling up some hints')
            # First, Get a map of the fake values
            self.mod.update()
            fill_value = -99
            fakevalues = np.array([fill_value for ij in range(self.data.voxelsBigSpace)])
            fakevalues = ma.masked_array(fakevalues, fakevalues==fill_value)
            counter = 0
            for index in self.indicescoarse:
                fakevalues[index] = zeehintmaker[counter]
                print('fake index hintmakerafter', fakevalues[index])
                counter += 1
            fakevalues = fakevalues.reshape(self.data.caseSide, self.data.caseSide)
            # Now fill up with guess values
            x, y = np.mgrid[0:fakevalues.shape[0], 0:fakevalues.shape[1]]
            xygood = np.array((x[~fakevalues.mask], y[~fakevalues.mask])).T
            xybad = np.array((x[fakevalues.mask], y[fakevalues.mask])).T
            fakevalues[fakevalues.mask] = fakevalues[~fakevalues.mask][KDTree(xygood).query(xybad)[1]]
            # Flatten the matrix to a 1 dimensional array.
            fakevalues = np.ravel(fakevalues)
            # write suions
            for counter, index in enumerate(self.indicesfine):
                # Assign the fake value as a hint.
                self.zeeVars[counter].setAttr("VarHintVal", fakevalues[index])
                # Assign a lower priority to hints that are not the real value from a previous iteration.
                if 10 != self.zeeVars[posnew].getAttr("VarHintPri"):
                    self.zeeVars[counter].setAttr("VarHintPri", 5)
        else:
            print('Nonexistent initial file')

    def launchOptimizationPWLOwnImplementation(self):
        print('creating VOI constraints and constraints directly associated with the objective...')
        self.objConstraintsPWLOwnImplementation()
        print('done')
        print('Setting up and launching the optimization...')
        # self.mod.write("modeltest.mps")
        self.loadprevioussolution()
        self.mod.optimize(mycallback)
        self.mod.write("solutionStep-" + str(self.data.sampleevery) + ".sol")
        print('done')

    def objConstraintsPWL(self):
        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on TARGETS
            if self.data.mask[i] in self.data.TARGETList:
                T = self.data.TARGETThresholds[np.where(self.data.mask[i] == self.data.TARGETList)[0]]
                points = [num for num in range(T - 1, T + 2)]
                self.mod.setPWLObj(self.zeeVars[i], points, [100, 0, 1])
            # Constraint on OARs
            elif self.data.mask[i] in self.data.OARList:
                T = self.data.OARThresholds[np.where(self.data.mask[i] == self.data.OARList)[0]]
                points = [num for num in range(T - 1, T + 2)]
                print("points:", points)
                self.mod.setPWLObj(self.zeeVars[i], points, [0, 0, 100])
            elif 0 == self.data.mask[i]:
                print('there is an element in the voxels that is also mask 0')
        ## sys.exit('there is a voxel that does not belong anywhere')
        # self.mod.addConstr(self.minDosePTVVar, grb.GRB.GREATER_EQUAL, 8.00)
        self.mod.update()

    def launchOptimizationPWL(self):
        print('creating VOI constraints and constraints directly associated with the objective...')
        self.objConstraintsPWL()
        print('done')
        print('Setting up and launching the optimization...')
        self.mod.write('pwlcrash.mps')
        self.mod.optimize()
        print('done')

    def objConstraintsQuadDose(self):
        self.overDoseConstraints = []
        self.underDoseConstraints = []
        self.overDoseVar = [None] * self.data.totalsmallvoxels
        self.underDoseVar = [None] * self.data.totalsmallvoxels
        for i in range(self.data.totalsmallvoxels):
            self.overDoseVar[i] = self.mod.addVar(lb = 0.0, ub=grb.GRB.INFINITY, vtype = grb.GRB.CONTINUOUS,
                                               name="overDoseVoxel" + str(i), column = None)
            self.underDoseVar[i] = self.mod.addVar(lb = 0.0, ub=grb.GRB.INFINITY, vtype = grb.GRB.CONTINUOUS,
                                               name="underDoseVoxel" + str(i), column = None)
        self.mod.update()
        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on
            if self.data.mask[i] in self.data.TARGETList:
                print('before error:', i, self.data.mask[i], self.data.TARGETList)
                print('failing expr.', np.where(self.data.mask[i]==self.data.TARGETList)[0])
                self.overDoseConstraints.append(self.mod.addConstr(self.overDoseVar[i], grb.GRB.GREATER_EQUAL,
                                                                  self.zeeVars[i] - self.data.TARGETThresholds[
                                                                       np.where(
                                                                           self.data.mask[i]==self.data.TARGETList)[0]]))

                self.underDoseConstraints.append(self.mod.addConstr(self.underDoseVar[i], grb.GRB.GREATER_EQUAL, 100 *(
                                                                  self.data.TARGETThresholds[np.where(
                                                                      self.data.mask[i] == self.data.TARGETList)[0]] -
                                                                    self.zeeVars[i])))
                # Constraint on OAR
            elif self.data.mask[i] in self.data.OARList:
                self.overDoseConstraints.append(self.mod.addConstr(self.overDoseVar[i], grb.GRB.GREATER_EQUAL, 100 * (
                                                                   self.zeeVars[i] - self.data.OARThresholds[
                                                                       np.where(
                                                                           self.data.mask[i] == self.data.OARList)[0]])))

                self.underDoseConstraints.append(self.mod.addConstr(self.underDoseVar[i], grb.GRB.GREATER_EQUAL,
                                                                self.data.OARThresholds[np.where(
                                                                    self.data.mask[i] == self.data.OARList)[0]] -
                                                                self.zeeVars[i]))
            elif 0 == self.data.mask[i]:
                print('there is an element in the voxels that is also mask 0')
                ## sys.exit('there is a voxel that does not belong anywhere')
        # self.mod.addConstr(self.minDosePTVVar, grb.GRB.GREATER_EQUAL, 8.00)
        self.mod.update()

    def launchOptimizationQuadratic(self):
        print('creating VOI constraints and constraints directly associated with the objective...')
        self.objConstraintsQuadDose()
        print('done')
        print('Setting up and launching the optimization...')

        self.objQuad = grb.QuadExpr()
        for i in range(self.data.totalsmallvoxels):
            self.objQuad += self.overDoseVar[i] * self.overDoseVar[i] + self.underDoseVar[i] * self.underDoseVar[i]

        self.mod.setObjective(self.objQuad, grb.GRB.MINIMIZE) #1.0 expresses minimization. It is the model sense.
        self.mod.update()
        self.mod.optimize()
        print('done')

    def launchOptimizationMaxMin(self):
        print('creating VOI constraints and constraints directly associated with the objective...')
        self.objConstraintsMinDose()
        print('done')
        print('Setting up and launching the optimization...')
        objexpr = grb.LinExpr()
        objexpr = self.minDosePTVVar

        self.mod.setObjective(objexpr, grb.GRB.MAXIMIZE) #1.0 expresses minimization. It is the model sense.
        self.mod.update()
        self.mod.optimize()
        print('done')

    def outputSolution(self):
        outDict = {}
        outDict['doseVector'] = np.array([self.zeeVars[j].X for j in range(self.data.totalsmallvoxels)])
        outDict['obj'] = self.mod.objVal

    def plotDVH(self, NameTag='', showPlot=False):
        voxDict = {}
        for t in self.data.TARGETList:
            voxDict[t] = np.where(self.data.mask == t)[0]
        for o in self.data.OARList:
            voxDict[o] = np.where(self.data.mask == o)[0]
        dose = np.array([self.zeeVars[j].X for j in range(self.data.totalsmallvoxels)])
        print(max(np.unique(dose)))
        print('dose: ', len(dose))
        plt.clf()
        for index, sValues in voxDict.items():
            sVoxels = sValues
            hist, bins = np.histogram(dose[sVoxels], bins=100)
            dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
            dvh = np.insert(dvh, 0, 1)
            plt.plot(bins, dvh, label = "struct " + str(index), linewidth = 2)

        lgd = plt.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title('DVH')
        plt.grid(True)
        plt.xlabel('Dose Gray')
        plt.ylabel('Fractional Volume')
        plt.savefig(self.data.outputDirectory + NameTag + '.png',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
        if showPlot:
            plt.show()
        plt.close()

    ## Showing the evolution of apertures through control points
    def plotSinoGram(self):
        ## Plotting apertures
        nrows, ncols = self.data.K, self.data.N
        image = -1 * np.ones(nrows * ncols)
        for k in range(self.data.K):
            for i in range(self.data.N):
                if 1 == self.binaryVars[i + k * self.data.N].X:
                    #print('beamlet '+str(i)+' in CP '+str(k)+ ' is open with intensity ', str(self.yVar[k].X))
                    # If this particular beamlet is open. Assign the intensity to it.
                    image[i + self.data.N * k] = self.yVar[k].X
        image = image.reshape((nrows, ncols))
        plt.clf()
        fig = plt.figure(1)
        cmapper = plt.get_cmap("autumn_r")
        norm = matplotlib.colors.Normalize(clip=False)
        cmapper.set_under('black')
        plt.imshow(image, cmap=cmapper, vmin=0.0, vmax=self.data.maxIntensity)
        plt.axis('off')
        plt.title('Sinogram with ' + str(self.data.sampleevery) + ' subsamples and ' + str(self.data.M) + ' event limit')
        fig.savefig(self.data.outputDirectory + 'sinogram.png', bbox_inches='tight')

    ## Show how many times you opened or closed each aperture.
    def plotEventsMU(self):
        ## DO NOT USE THIS FUNCTION> USE plotEventsbinary instead
        arrayofevents = []
        for i in range(self.data.N):
            arrayofevents.append(0)
            for k in range(self.data.K - 1):
                arrayofevents[-1] += self.muVars[i + k * self.data.N].X
        ind = range(len(arrayofevents))
        plt.clf()
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, arrayofevents, 0.6, color='r')
        plt.title('Events per beamlet')
        fig.savefig(self.data.outputDirectory + 'beamletdistributionMU.png', bbox_inches='tight')

    def plotEventsbinary(self):
        arrayofevents = []
        for i in range(self.data.N):
            arrayofevents.append(0)
            for k in range(self.data.K - 1):
                arrayofevents[-1] += abs(self.binaryVars[i + k * self.data.N].X - self.binaryVars[i + (k + 1) * self.data.N].X)
        ind = range(len(arrayofevents))
        plt.clf()
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, arrayofevents, 0.6, color='r')
        plt.title('Events per beamlet')
        fig.savefig(self.data.outputDirectory + 'beamletdistribution.png', bbox_inches='tight')

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
        self.outputDirectory = "output/"
        ## M value. Number of times per beamlet that the switch can be turned on or off
        self.M = 10
        ## C Value in the objective function
        self.C = 1.0
        ## ry this number of observations
        self.coarse = 32
        self.sampleevery = 16
        ## N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.N = 80
        self.maxIntensity = 1000
        self.maxDosePTV = 999.9
        ## Number of control points (every 2 degrees)
        self.K = 178
        ## Total number of beamlets
        ## OARMAX is maximum dose tolerable for organs. 10 in this case
        self.OARMAX = 7.0
        print('Read vectors...')
        #self.readWilmersCase()
        self.readWeiguosCase()
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        print('Build sparse matrix.')
        # The next part uses the case corresponding to either Wilmer or Weiguo's case.
        self.totalbeamlets = self.K * self.N
        self.totalsmallvoxels = max(self.smallvoxels) + 1
        print('totalsmallvoxels: ', self.totalsmallvoxels)
        print('a brief description of Dijs array', describe(self.Dijs))
        self.D = sps.csc_matrix((self.Dijs, (self.smallvoxels, self.bixels)), shape=(self.totalsmallvoxels, self.totalbeamlets))
        ## This is the total number of voxels that there are in the body. Not all voxels from all directions

    ## Create a map from big to small voxel space
    def BigToSmallCreator(self):
        # Notice that the order of voxels IS preserved. So (1,2,3,80,7) produces c = (0,1,2,4,3)
        a, b, c, d = np.unique(self.voxels, return_index=True, return_inverse=True, return_counts=True)
        return(c)

    ## This function reduces the size of the case by a factor fr
    def sizereduction(self, fr):
        maxlimit = max(self.voxels)
        print('MAXLIMIT: ', maxlimit)
        existingvoxels = np.arange(0, maxlimit, fr)
        indices = np.where(np.in1d(self.voxels, existingvoxels))[0]
        self.bixels = self.bixels[indices]
        self.voxels = self.voxels[indices]
        self.Dijs = self.Dijs[indices]
        self.mask = np.array([self.mask[i] for i in np.arange(0, maxlimit, fr)])
        print('reduced size of mask', len(self.mask))
        print('reduced size of voxels', len(self.voxels))
        print('reduced size of Dijs', len(self.Dijs))

    ## Choose Small Space
    def chooseSmallSpace(self, stepsparse):
        # Original Map
        om = [ij for ij in range(self.voxelsBigSpace)]
        # New Map
        nm = []
        for i in np.arange(0, self.caseSide, stepsparse):
            for j in np.arange(0, self.caseSide, stepsparse):
                nm.append(om[int(j) + int(i) * self.caseSide])
        # Summary statistics of voxels
        print('om', describe(om))
        print('nm', describe(nm))
        print('effectivity of the reduction:', len(om)/len(nm))
        indices = np.where(np.in1d(self.voxels, nm))[0]
        self.bixels = self.bixels[indices]
        self.voxels = self.voxels[indices]
        self.Dijs = self.Dijs[indices]
        print(len(self.mask))
        self.mask = np.array([self.mask[i] for i in nm])

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        self.bixels = getvector('data\\Bixels_out.bin', np.int32)
        self.voxels = getvector('data\\Voxels_out.bin', np.int32)
        print('voxels what it looks like:', self.voxels)
        self.Dijs = getvector('data\\Dijs_out.bin', np.float32)
        self.mask = getvector('data\\optmask.img', np.int32)
        self.caseSide = 256
        self.voxelsBigSpace = self.caseSide ** 2
        print('bixels length: ', len(self.bixels))
        print('VOXELS length: ', len(self.voxels))
        print('dijs length: ', len(self.Dijs))
        print('mask length: ', len(self.mask))
        print('remove all values of zeroes... and double checking')
        # First keep a copy of the high definition space
        self.bixelsHD = self.bixels
        self.voxelsHD = self.voxels
        self.DijsHD = self.Dijs
        self.maskHD = self.mask

        # if os.path.isfile("solutionStep-" + str(self.coarse) + ".sol"):
        #     print('Existent initial File')
        #     self.compareHDandSmallSpace(self.coarse, self.sampleevery)
        # else:
        #     print('Nonexistent initial File')

        self.chooseSmallSpace(self.sampleevery)
        # Next I am removing the voxels that have a mask of zero (0)
        locats = np.where(0 == self.mask)[0]
        self.mask = np.delete(self.mask, locats)
        print('locats: ', len(locats))
        print('mask length: ', len(self.mask))
        indices = np.where(np.in1d(self.voxels, locats))[0]
        print('indices', indices)
        self.bixels = np.delete(self.bixels, indices)
        self.voxels = np.delete(self.voxels, indices)
        self.Dijs = np.delete(self.Dijs, indices)
        print('bixels out length: ', len(self.bixels))
        print('VOXELS out length: ', len(self.voxels))
        print('unique voxel elements:', len(np.unique(self.voxels)))
        print('dijs out length: ', len(self.Dijs))
        print('mask out length: ', len(self.mask))
        self.OARList = [1, 2, 3]
        self.OARThresholds = [7, 8, 9]
        self.TARGETList = [256]
        self.TARGETThresholds = [14]
        #print(np.unique(self.mask))

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

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
start_time = time.time()
dataobject = tomodata()
tomoinstance = tomotherapyNP(dataobject)
print("--- %s seconds ---" % (time.time() - start_time))
