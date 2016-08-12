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
# Original template from Troy Long.
class tomotherapyNP(object):
    def __init__(self, datastructure):
        print('Reading in data...')
        self.data = datastructure
        print('Building column generation method')
        self.thresholds()
        self.rmpres = self.ColumnGenerationMain()
        print('The problem has been completed')

    def thresholds(self):
        self.quadHelperThresh = [None] * self.data.totalsmallvoxels
        self.quadHelperOver = [None] * self.data.totalsmallvoxels
        self.quadHelperUnder = [None] * self.data.totalsmallvoxels
        for i in range(0, self.data.totalsmallvoxels):
            # Constraint on TARGETS
            T = None
            if self.data.mask[i] in self.data.TARGETList:
                T = self.data.TARGETThresholds[np.where(self.data.mask[i] == self.data.TARGETList)[0][0]]
                self.quadHelperOver[i] = 1.0
                self.quadHelperUnder[i] = 1000.0
            # Constraint on OARs
            elif self.data.mask[i] in self.data.OARList:
                T = self.data.OARThresholds[np.where(self.data.mask[i] == self.data.OARList)[0][0]]
                self.quadHelperOver[i] = 100.0
                self.quadHelperUnder[i] = 0.0
            elif 0 == self.data.mask[i]:
                print('there is an element in the voxels that is also mask 0')
            self.quadHelperThresh[i] = T

    def calcDose(self):
        # Remember. The '*' operator is elementwise multiplication. Here, ordered by beamlets first, then control points
        intensities = (self.yk * self.binaryVariables.T).T.reshape(self.data.K * self.data.N, 1)
        self.currentDose = np.asarray(self.data.D.dot(intensities)) # conversion to array necessary. O/W algebra wrong
        # The line below effectively multiplies each element in data.D by one or zero. USE THE DENSE VERSION OF D.
        matdzdk = np.multiply(self.data.Ddense, self.binaryVariables)
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

    def solveRMC(self):
        self.calcObjGrad(self.yk)
        # Create the boundaries of intensities
        print(type(self.yk.tolist()))
        res = minimize(self.calcObjGrad, self.yk, method='L-BFGS-B', jac=True, bounds=self.boundschoice, options={'ftol': 1e-3, 'disp': 5, 'maxiter': 200})
        print('exiting graciously')
        return (res)

    # Define the variables
    def addVariables(self, b):
        self.binaries = [None] * self.data.K
        self.muVars = [None] * (self.data.K - 1)
        for i in range(0, self.data.K):
            self.binaries[i] = self.mod.addVar(vtype=grb.GRB.BINARY,
                                               name="binaryVoxel_{" + str(i) + ", " + str(b) + "}",
                                               column=None)
            #  The mu variable will register change in the behaviour from one control to the other. Therefore loses 1
            # degree of freedom
            if (self.data.K - 1) != i:
                self.muVars[i] = self.mod.addVar(vtype=grb.GRB.BINARY,
                                                 name="mu_{" + str(i) + "," + str(b) + "}",
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
        self.mod.addConstr(expr, grb.GRB.LESS_EQUAL, self.data.M, name="summaxrest_{" + str(k) + "}")
        self.mod.update()

    def addGoal(self, b):
        expr = grb.LinExpr()
        for i in range(self.data.K):
            expr += self.binaries[i] * self.doseandgradient[b + i * self.data.N, 0]
            # Use this opportunity to initialize the binary variables to something being closed. So that way I will
            # always close the beamlets instead of having them open. This may slow things down.
            # self.binaries[i].Start = 0
        self.mod.update()
        self.mod.setObjective(expr, grb.GRB.MINIMIZE)

    def PPoptimization(self, b):
        # The idea is that for each beamlet I will produce a set of length K of apertures
        self.mod = grb.Model()
        # self.mod.params.OutputFlag = 1
        self.mod.params.threads = 1
        self.mod.params.MIPFocus = 1
        self.mod.params.Presolve = 0
        # self.mod.params.TimeLimit = 14 * 3600.0
        self.addVariables(b)
        self.addConstraints(b)
        self.addGoal(b)
        self.mod.optimize()
        # Get the optimal value of the optimization
        if self.mod.status == grb.GRB.Status.OPTIMAL:
            obj = self.mod.getObjective()
            self.goalvalues[b] = obj.getValue()
            # Assign optimal results of the optimization
            for i in range(self.data.K):
                self.goaltargets[i, b] = self.binaries[i].X

    ## Solves the pricing problem as set up
    def PricingProblem(self):
        # Prepare the matrix multiplying all the rows times the $\partial F / \partial z_j$ vector
        print('DT shape', self.data.D.T.shape)
        print('voxelgradient shape', self.voxelgradient.shape)
        self.doseandgradient = -self.data.D.T * self.voxelgradient.T
        # Run an optimization problem for each of the different beamlets available (those that don't let light in)
        candidatebeamlets = np.where(1 == self.mathCal)[0].tolist()
        self.goalvalues = np.array([np.inf] * self.data.N)
        self.goaltargets = np.ones(self.data.K, self.data.N) * None
        iterpricingprob = 1
        for b in candidatebeamlets:
            print('iteration of pricing problem', iterpricingprob, 'out of ', len(candidatebeamlets),
                  ' trying candidate beamlet ', str(b))
            self.PPoptimization(b)

        bestbeamlet = np.argmin(self.goalvalues)
        bestgoal = self.goalvalues[bestbeamlet]
        print('value of best goal was', bestgoal)
        # For each of the beamlets. Assign the resulting path to the matrix of binaryVariables if bestgoal < 0
        if bestgoal <= 0.0:
            self.mathCal[bestbeamlet] = 0
            for i in range(self.data.K):
                # Update the beamlets available
                self.binaryVariables[i, bestbeamlet] = self.goaltargets[i, bestbeamlet]
        return(bestgoal)

    ## This function creates a matrix that has a column of ones kronecker the identity matrix
    def onehelpCreator(self):
        self.oneshelper = np.zeros((self.data.N * self.data.K , self.data.K))
        for i in range(self.data.K):
            for j in range(self.data.N):
                self.oneshelper[j + i * self.data.N, i] = 1.0

    def ColumnGenerationMain(self):
        # Create the ones helper matrix:
        self.onehelpCreator()
        # Step 1: Assign \mathcal{C} the empty set. Remember to open everytime I add a path.
        self.mathCal = np.ones(self.data.N, dtype=np.int)
        # Matrix with a binary choice for each of the beamlets at each control point.
        self.binaryVariables = np.ones((self.data.K, self.data.N))
        # Variable that keeps the intensities at each control point. Initialized at maximum intensity
        self.yk = np.ones(self.data.K) * self.data.maxIntensity
        # Calculate the boundaries of yk
        self.boundschoice = [(0, self.data.maxIntensity),] * self.data.K
        gstar = -np.inf
        iterCG = 1
        rmpres = None
        while (gstar <= 0) & ( sum(1 - self.mathCal) < self.data.N) & (time.time() - start_time < 500):
            print('starting iteration of column generation', iterCG)
            self.plotSinoGram(iterCG)
            iterCG += 1
            # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an
            # instance of the PP
            self.calcDose()
            self.calcGradientandObjValue()
            gstar = self.PricingProblem()
            # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal
            # solution to the PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
            if gstar >= 0:
                # This choice includes the case when no aperture was selected
                print('Program finishes because no beamlet was selected to enter')
                break
            else:
                rmpres = self.solveRMC(  )
        print('leaving CG')
        return(rmpres)

    def plotDVH(self, NameTag='', showPlot=False):
        voxDict = {}
        for t in self.data.TARGETList:
            voxDict[t] = np.where(self.data.mask == t)[0]
        for o in self.data.OARList:
            voxDict[o] = np.where(self.data.mask == o)[0]
        if self.rmpres is None:
            print('did I enter here?')
            sys.stderr.write('the master problem does not had a valid solution')
        dose = np.array([self.currentDose[j] for j in range(self.data.totalsmallvoxels)])
        plt.clf()
        for index, sValues in voxDict.items():
            sVoxels = sValues
            hist, bins = np.histogram(dose[sVoxels], bins=100)
            dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
            dvh = np.insert(dvh, 0, 1)
            plt.plot(bins, dvh, label = "struct " + str(index), linewidth = 2)

        lgd = plt.legend(fancybox = True, framealpha = 0.5, bbox_to_anchor = (1.05, 1), loc = 2)
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
    def plotSinoGram(self, thisname=None):
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
                arrayofevents[-1] += abs(self.binaryVariables[k, i] - self.binaryVariables[k, i])
        ind = range(len(arrayofevents))
        plt.clf()
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, arrayofevents, 0.6, color='r')
        plt.title('Events per beamlet')
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
        ## M value. Number of times per beamlet that the switch can be turned on or off
        self.M = 180
        ## C Value in the objective function
        self.C = 1.0
        ## ry this number of observations
        self.sampleevery = 25
        self.coarse = self.sampleevery * 2
        ## N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.N = 80
        self.maxIntensity = 20
        self.caseSide = 256
        self.voxelsBigSpace = self.caseSide ** 2
        ## Number of control points (every 2 degrees)
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
        #-------------------------------------
        locats = np.where(0 == self.maskHD)[0]
        self.maskHD = np.delete(self.maskHD, locats)
        indices = np.where(np.in1d(self.voxelsHD, locats))[0]
        self.bixelsHD = np.delete(self.bixelsHD, indices)
        self.voxelsHD = np.delete(self.voxelsHD, indices)
        self.DijsHD = np.delete(self.DijsHD, indices)

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        self.bixels = getvector('data\\Bixels_out.bin', np.int32)
        self.voxels = getvector('data\\Voxels_out.bin', np.int32)
        self.Dijs = getvector('data\\Dijs_out.bin', np.float32)
        self.mask = getvector('data\\optmask.img', np.int32)

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
        self.OARThresholds = [7, 8, 9]
        self.TARGETList = [256]
        self.TARGETThresholds = [14]

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
start_time = time.time()
dataobject = tomodata()
tomoinstance = tomotherapyNP(dataobject)
tomoinstance.plotDVH('dvh-ColumnGeneration')
tomoinstance.plotSinoGram()
tomoinstance.plotEventsbinary()
print("--- %s seconds ---" % (time.time() - start_time))
