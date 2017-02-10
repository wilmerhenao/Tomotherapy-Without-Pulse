__author__ = 'wilmer'

try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import gurobipy as grb
import pandas as pds
from scipy.optimize import minimize
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sps
import time
from scipy.stats import describe
import pickle
import matplotlib

numcores = 4

def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        mipgap = 0.5 # Here 0.2 means 2%
        if 0 != objbst:
            if abs(objbst - objbnd)/abs(objbst) < mipgap:
                print('Stop early -', str(mipgap * 100), '% gap achieved')
                model.terminate()

## Class definition of the gurobi object that handles creation and execution of the model
class tomotherapyNP(object):
    def __init__(self, datastructure):
        print('Reading in data...')
        self.data = datastructure
        print('Building column generation method')
        self.thresholds()
        self.rmpres = self.IterativeMain()
        print('The problem has been completed')

    def thresholds(self):
        self.quadHelperThresh = self.data.quadHelperThresh
        self.quadHelperOver = self.data.quadHelperOver
        self.quadHelperUnder = self.data.quadHelperUnder

    def calcDose(self):
        # Remember. The '*' operator is elementwise multiplication. Here, ordered by beamlets first, then control points
        intensities = (self.yk * self.binaryVariables.T).T.reshape(self.data.K * self.data.N, 1)
        self.currentDose = np.asarray(self.data.D.dot(intensities)) # conversion to array necessary. O/W algebra wrong
        # The line below effectively multiplies each element in data.D by one or zero. USE THE DENSE VERSION OF D
        matdzdk = np.multiply(self.data.Ddense, np.tile(self.binaryVariables.reshape(1, self.data.K * self.data.N), (self.data.totalsmallvoxels ,1)))
        # Oneshelper adds in the creation of the dzdk it is a matrix of ones that adds the right positions in matdzdk
        self.dZdK = np.dot(matdzdk, self.oneshelper)
        # Assert the tuple below
        assert((self.data.totalsmallvoxels, self.data.K) == self.dZdK.shape)

    ## This function regularly enters the optimization engine to calculate objective function and gradients
    def calcGradientandObjValue(self):
        oDoseObj = self.currentDose.T - self.quadHelperThresh
        oDoseObjCl = (oDoseObj > 0) * oDoseObj
        oDoseObj = oDoseObjCl * oDoseObjCl * self.quadHelperOver
        uDoseObj = self.quadHelperThresh - self.currentDose.T
        uDoseObjCl = (uDoseObj > 0) * uDoseObj
        uDoseObj = uDoseObjCl * uDoseObjCl * self.quadHelperUnder
        self.objectiveValue = np.sum(oDoseObj + uDoseObj)
        oDoseObjGl = 2 * oDoseObjCl * self.quadHelperOver
        uDoseObjGl = 2 * uDoseObjCl * self.quadHelperUnder
        # Notice that I use two types of gradients.
        # One for voxels and one for apertures. The apertures one will be
        # sent to the optimizer
        self.voxelgradient = 2 * (oDoseObjGl - uDoseObjGl) # With respect to doses.
        self.aperturegradient = (self.voxelgradient * self.dZdK).tolist()[0] # If not, then fortran won't understand
        assert(self.voxelgradient.shape ==  (1, self.data.totalsmallvoxels))
        assert(len(self.aperturegradient) == self.data.K)

    def calcObjGrad(self, x, user_data=None):
        self.yk = x
        self.calcDose()
        self.calcGradientandObjValue()
        #print('types', type(self.objectiveValue), self.aperturegradient.shape)
        return(self.objectiveValue, np.array(self.aperturegradient))

    def solveRMC(self, precision = 1e-2):
        self.calcObjGrad(self.yk)
        res = minimize(self.calcObjGrad, self.yk, method='L-BFGS-B', jac=True, bounds=self.boundschoice, options={'ftol': precision, 'disp': 5, 'maxiter': 200})
        self.yk = res.x
        self.calcObjGrad(self.yk)
        print('exiting graciously')
        return (res)

    # Define the variables
    def addVariables(self):
        self.binaries = [None] * self.data.K * self.data.N
        self.muVars = [None] * (self.data.K - 1) * self.data.N
        self.zjs = [None] * (len(self.zbar))
        self.xiplus = [None] * (len(self.zbar))
        self.ximinus = [None] * (len(self.zbar))
        for i in range(0, self.data.K):
            for b in range(0, self.data.N):
                self.binaries[i + b * self.data.K] = self.mod.addVar(vtype=grb.GRB.BINARY,
                                                   name="binaryVoxel_{" + str(i) + ", " + str(b) + "}",
                                                   column=None)
                #  The mu variable will register change in the behaviour from one control to the other. Therefore loses 1
                # degree of freedom
                if (self.data.K - 1) != i:
                    self.muVars[i + b * (self.data.K - 1)] = self.mod.addVar(vtype=grb.GRB.BINARY,
                                                                             name="mu_{" + str(i) + "," + str(b) + "}",
                                                                             column=None)
        for i in range(0, len(self.zbar)):
            self.zjs[i] = self.mod.addVar(vtype=grb.GRB.CONTINUOUS,
                                          name="dose_{" + str(i) + "}",
                                          column=None, lb=0.0)
            self.xiplus[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="xiplus_{" + str(i) + "}",
                                             column=None)
            self.ximinus[i] = self.mod.addVar(lb=0.0, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS,
                                             name="ximinus_{" + str(i) + "}",
                                             column=None)

        self.mod.update()

    def addConstraints(self, b):
        # Add some constraints. This one is about replacing the absolute value with linear expressions
        self.absoluteValueRemovalConstraint1 = [None] * (self.data.K - 1)
        self.absoluteValueRemovalConstraint2 = [None] * (self.data.K - 1)
        # self.sumMaxRestriction = None
        expr = grb.LinExpr()
        # Constraints related to absolute value removal from objective function
        for k in range(0, (self.data.K - 1)):
            # sum mu variables and restrict their sum to be smaller than M
            expr += self.muVars[k]
            # \mu \geq \beta_{i,k+1} - \beta_{i,k}
            self.absoluteValueRemovalConstraint1[k] = self.mod.addConstr(
                self.muVars[k], grb.GRB.GREATER_EQUAL, self.binaries[k + 1] - self.binaries[k],
                name="rmabs1_{" + str(k) + "," + str(b) + "}")
            # \mu \geq -(\beta_{i,k+1} - \beta_{i,k})
            self.absoluteValueRemovalConstraint2[k] = self.mod.addConstr(
                self.muVars[k], grb.GRB.GREATER_EQUAL, -(self.binaries[k + 1] - self.binaries[k]),
                name="rmabs2_{" + str(k) + "," + str(b) + "}")
        lastconstr = self.mod.addConstr(expr, grb.GRB.LESS_EQUAL, self.data.M, name="summaxrest_{" + str(k) + "}")
        # Constraints related to dose.
        # First of all create a submatrix of sparse D.
        indices = []
        for i in range(0, self.data.K):
            indices.append(self.data.N * i + b)
        restrictedIntensity = (self.data.D.tocsc()[:,indices])
        # Multiply times the intensity in each of the different columns (this is y bar)
        for i in range(0, self.data.K):
            restrictedIntensity[:,i] *= self.yk[i]

        for j in range(0, len(self.zjs)):
            expr = grb.LinExpr()
            expr += self.zjs[j] - self.zbar[j]
            for k in range(0, (self.data.K - 1)):
                expr -= restrictedIntensity[j, k]  * self.binaries[k]
            self.mod.addConstr(expr, grb.GRB.EQUAL, 0, name="z_j{" + str(j) + "}")
            self.mod.addConstr(self.zjs[j] - self.quadHelperThresh[j], grb.GRB.EQUAL, self.xiplus[j] - self.ximinus[j], name="xis{" + str(j) + "}")
        self.mod.update()

    def addGoalExact(self, b):
        expr = grb.QuadExpr()
        for i in range(0,len(self.zjs)):
            expr += self.quadHelperOver[i] * self.xiplus[i] * self.xiplus[i] + self.quadHelperUnder[i] * self.ximinus[i] * self.ximinus[i]
        self.mod.setObjective(expr, grb.GRB.MINIMIZE)
        self.mod.update()

    def PricingProblem(self):
        # The idea is that for each beamlet I will produce a set of length K of apertures
        self.mod = grb.Model()
        # Fix a vector z bar as explained on equation 21 in the document
        self.zbar = 0
        self.calcDose()
        self.zbar = self.currentDose
        self.addVariables()
        self.addConstraints()
        self.addGoalExact()
        self.mod.optimize()
        # Get the optimal value of the optimization
        if self.mod.status == grb.GRB.Status.OPTIMAL:
            obj = self.mod.getObjective()
            self.goalvalues[b] = obj.getValue()
            # Assign optimal results of the optimization
            for i in range(self.data.K):
                self.goaltargets[i, b] = self.binaries[i].X

    ## This function creates a matrix that has a column of ones kronecker the identity matrix
    def onehelpCreator(self):
        self.oneshelper = np.zeros((self.data.N * self.data.K , self.data.K))
        for i in range(self.data.K):
            for j in range(self.data.N):
                self.oneshelper[j + i * self.data.N, i] = 1.0

    def refinesolution(self, iterCG):
        gstar = -np.inf
        numrefinements = 1
        while (gstar <= 0) & ( numrefinements < 100):
            print('starting iteration of column generation', iterCG)
            # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an
            # instance of the PP
            numrefinements += 1
            self.mathCal = np.array([i for i in self.originalMathCal])
            gstar = self.PricingProblem()
            iterCG += 1
            print('this is refinement' + str(iterCG))
            self.plotSinoGram(iterCG)
            # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal
            # solution to the PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
            if gstar >= 0:
                # This choice includes the case when no aperture was selected
                print('Program finishes because no beamlet was selected to enter')
                break
            else:
                self.rmpres = self.solveRMC()
                self.plotDVH('dvh-ColumnGeneration' + str(iterCG))
        print('leaving solution refinement')

    ## Find the locations where the D matrix will just never reach and don't even look at those beamlets
    def turnOnOnlynecessarybeamlets(self):
        for i in np.unique(self.data.bixels % self.data.N):
            self.mathCal[i] = 1

    def IterativeMain(self):
        # Create the ones helper matrix:
        self.onehelpCreator()

        # Matrix with a binary choice for each of the beamlets at each control point
        self.binaryVariables = np.zeros((self.data.K, self.data.N))
        # Turn off unnecessary beamlest to save time
        self.turnOnOnlynecessarybeamlets()
        # Variable that keeps the intensities at each control point. Initialized at maximum intensity
        self.yk = np.ones(self.data.K) * self.data.maxIntensity
        # Calculate the boundaries of yk
        self.boundschoice = [(0, self.data.maxIntensity),] * self.data.K
        oldobj = np.Inf
        newobj = np.Inf
        iterCG = 1
        self.rmpres = None
        while (True):
            oldobj = newobj
            print('starting iteration of column generation', iterCG)
            self.plotSinoGram(iterCG)
            # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an
            # instance of the PP
            self.calcObjGrad(self.yk)
            gstar = self.PricingProblem()
            # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal
            # solution to the PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
            self.rmpres = self.solveRMC()
            newobj = self.rmpres.fun
            self.plotDVH('dvh-ColumnGeneration' + str(iterCG))

            if(abs(newobj - oldobj)/oldobj < 0.01)
                break
            else
                iterCG += 1
        print('starting solution refinement')
        self.refinesolution(iterCG)
        self.rmpres = self.solveRMC(1E-5)
        print('leaving CG')
        return(self.rmpres)

    # Plot the dose volume histogram
    def plotDVH(self, NameTag='', showPlot=False):
        voxDict = {}
        for t in self.data.TARGETList:
            voxDict[t] = np.where(self.data.mask == t)[0]
        for o in self.data.OARList:
            voxDict[o] = np.where(self.data.mask == o)[0]
        if self.rmpres is None:
            print('did I enter here?')
            sys.stderr.write('the master problem does not have a valid solution')
        dose = np.array([self.currentDose[j] for j in range(self.data.totalsmallvoxels)])
        plt.clf()
        for index, sValues in voxDict.items():
            sVoxels = sValues
            hist, bins = np.histogram(dose[sVoxels], bins=100)
            dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
            dvh = np.insert(dvh, 0, 1)
            plt.plot(bins, dvh, label="struct " + str(index), linewidth=2)
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
    def plotSinoGram(self, thisname=""):
        ## Plotting apertures
        nrows, ncols = self.data.K, self.data.N
        image = -1 * np.ones((nrows, ncols))
        for k in range(self.data.K):
            for i in range(self.data.N):
                if 1 == self.binaryVariables[k, i]:
                    # If this particular beamlet is open. Assign the intensity to it.
                    image[k, i] = self.yk[k]
        plt.clf()
        fig = plt.figure(1)
        cmapper = plt.get_cmap("autumn_r")
        cmapper.set_under('black')
        plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = self.data.maxIntensity)
        plt.title('Sinogram subsamples = ' + str(self.data.sampleevery) + ' and ' + str(self.data.M) + ' events limit')
        plt.xlabel('Beamlets')
        plt.ylabel('Control Points')
        fig.savefig(self.data.outputDirectory + 'sinogram' + str(thisname) + '.png', bbox_inches='tight')

    def plotEventsbinary(self):
        arrayofevents = []
        for i in range(self.data.N):
            arrayofevents.append(0)
            for k in range(self.data.K - 1):
                arrayofevents[-1] += abs(self.binaryVariables[k+1, i] - self.binaryVariables[k, i])
        ind = range(len(arrayofevents))
        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(ind, arrayofevents, 0.6, color='r')
        plt.title('Events per beamlet')
        print(arrayofevents)
        fig.savefig(self.data.outputDirectory + 'beamletdistribution.png', bbox_inches='tight')

## Function that reads the files produced by Weiguo
def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)

class tomodata:
    ## Initialization of the data
    def __init__(self):
        self.outputDirectory = "output/"
        # M value. Number of times per beamlet that the switch can be turned on or off
        self.M = 50
        # C Value in the objective function
        self.C = 1.0
        # ry this number of observat
        # ions
        self.sampleevery = 32
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.N = 80
        self.maxIntensity = 10
        self.caseSide = 256
        self.voxelsBigSpace = self.caseSide ** 2
        # Number of control points (every 2 degrees)
        self.K = 178
        print('Read vectors...')
        self.readWeiguosCase(  )
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        print('Build sparse matrix.')
        # The next part uses the case corresponding to either Wilmer or Weiguo's case
        self.totalbeamlets = self.K * self.N
        self.totalsmallvoxels = max(self.smallvoxels) + 1
        print('totalsmallvoxels:', self.totalsmallvoxels)
        print('a brief description of Dijs array', describe(self.Dijs))
        self.D = sps.csr_matrix((self.Dijs, (self.smallvoxels, self.bixels)), shape=(self.totalsmallvoxels, self.totalbeamlets))
        self.Ddense = self.D.todense()
        self.quadHelperThresh = np.zeros(len(self.mask))
        self.quadHelperUnder = np.zeros(len(self.mask))
        self.quadHelperOver = np.zeros(len(self.mask))
        #######################################3
        for i in range(len(self.mask)):
            # Constraint on TARGETS
            T = None
            if self.mask[i] in self.TARGETList:
                T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                self.quadHelperOver[i] = 0.01
                self.quadHelperUnder[i] = 0.6
            # Constraint on OARs
            elif self.mask[i] in self.OARList:
                T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                self.quadHelperOver[i] = 0.001
                self.quadHelperUnder[i] = 0.00
            elif 0 == self.mask[i]:
                print('there is an element in the voxels that is also mask 0')
            self.quadHelperThresh[i] = T
            ########################
        ## This is the total number of voxels that there are in the body. Not all voxels from all directions

    ## Create a map from big to small voxel space, the order of elements is preserved but there is a compression to only
    # one element in between.
    def BigToSmallCreator(self):
        # Notice that the order of voxels IS preserved. So (1,2,3,80,7) produces c = (0,1,2,4,3)
        a, b, c, d = np.unique(self.voxels, return_index=True, return_inverse=True, return_counts=True)
        print('BigToSmallCreator:size of c. Size of the problem:', len(c))
        return(c)

    ## Choose Small Space is the function that reduces the resolution to work with. voxelsHD will remain to be used for
    # functions that require high resolution. Mainly when loading the hints.
    def chooseSmallSpace(self, stepsparse):
        # Original Map
        om = [ij for ij in range(self.voxelsBigSpace)]
        # New Map
        nm = []
        for i in np.arange(0, self.caseSide, stepsparse):
            for j in np.arange(0, self.caseSide, stepsparse):
                nm.append(om[int(j) + int(i) * self.caseSide])
        # Summary statistics of voxels
        print('effectivity of the reduction:', len(om)/len(nm))
        indices = np.where(np.in1d(self.voxels, nm))[0]
        self.bixels = self.bixels[indices]
        self.voxels = self.voxels[indices]
        self.Dijs = self.Dijs[indices]
        print(len(self.mask))
        self.mask = np.array([self.mask[i] for i in nm])
        locats = np.where(0 == self.mask)[0]
        self.mask = np.delete(self.mask, locats)

    def removezeroes(self):
        # Next I am removing the voxels that have a mask of zero (0) because they REALLY complicate things otherwise
        # Making the problem larger.
        #-------------------------------------
        locats = np.where(0 == self.maskHD)[0]
        self.maskHD = np.delete(self.maskHD, locats)
        indices = np.where(np.in1d(self.voxelsHD, locats))[0]
        self.bixelsHD = np.delete(self.bixelsHD, indices)
        self.voxelsHD = np.delete(self.voxelsHD, indices)
        self.DijsHD = np.delete(self.DijsHD, indices)

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        self.bixels = getvector('data/Bixels_out.bin', np.int32)
        self.voxels = getvector('data/Voxels_out.bin', np.int32)
        self.Dijs = getvector('data/Dijs_out.bin', np.float32)
        self.mask = getvector('data/optmask.img', np.int32)

        # First keep a copy of the high definition
        self.bixelsHD = self.bixels
        self.voxelsHD = self.voxels
        self.DijsHD = self.Dijs
        self.maskHD = self.mask

        # Choose a smaller space (a subsample). And remove zeroes from the map since these are unimportant structures.
        self.chooseSmallSpace(self.sampleevery)
        self.removezeroes()

        # Assign structures and thresholds for each of them
        self.OARList = [1, 2, 3]
        self.OARThresholds = [10, 15, 20]
        self.TARGETList = [256]
        self.TARGETThresholds = [70]

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
## This part is for AMPL's implementation:
def printAMPLfile(data):
    f = open("tomononlinear.dat", "w")
    print('param numvoxels :=', len(data.mask), ';', file = f)
    print('param: VOXELS: thethreshold :=', file = f)
    thrs = pds.DataFrame(data = {'A': np.arange(len(data.mask)), 'B': data.quadHelperThresh})
    print(thrs.to_string(index=False, header = False), file = f)
    print(";", file=f)
    print('param: quadHelperOver :=', file = f)
    thrs = pds.DataFrame(data = {'A': np.arange(len(data.mask)), 'B': data.quadHelperOver})
    print(thrs.to_string(index=False, header = False), file = f)
    print(";", file=f)
    print('param: quadHelperUnder :=', file=f)
    thrs = pds.DataFrame(data={'A': np.arange(len(data.mask)), 'B': data.quadHelperUnder})
    print(thrs.to_string(index=False, header = False), file=f)
    print(";", file=f)
    print('param: KNJPARAMETERS: D:=' , file = f)
    leafs = (data.bixels % data.N).astype(int)
    projections = np.floor(data.bixels / data.N).astype(int)
    sparseinfo = pds.DataFrame(data = {'LEAVES' : leafs, 'PROJECTIONS' : projections, 'VOXELS' : data.smallvoxels, 'ZDOSES' : data.Dijs})
    print(sparseinfo.to_string(index=False, header=False), file = f)
    print(";", file = f)
    f.close()

start_time = time.time()
dataobject = tomodata()
printAMPLfile(dataobject)
tomoinstance = tomotherapyNP(dataobject)
tomoinstance.plotDVH('dvh-ColumnGeneration')
tomoinstance.plotSinoGram()
tomoinstance.plotEventsbinary()
print("--- %s seconds ---" % (time.time() - start_time))
