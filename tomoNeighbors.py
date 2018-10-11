__author__ = 'wilmer'

try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe
import subprocess
import math
from gurobipy import *
from collections import defaultdict
import pprint

# User input goes here and only here
numcores = 8
initialProjections = 51
numberOfLeaves = 80
maxvoxels = 2000
tumorsite = "Prostate"
timeko = 0.1 # secs
timekc = 0.095 # secs
time10 = 10
speed = 24 # degrees per second
howmanydegreesko = speed * timeko # Degrees spanned in this time
howmanydegreeskc = speed * timekc # Degrees spanned in this time
howmanydegrees10 = speed * time10 # Degrees spanned in 10 seconds
t51 = (360/initialProjections) / speed
delta51 = speed * t51 # Espacio es igual a velocidad por tiempo
subdivisions = max(math.ceil(t51/timeko), math.ceil(t51/timekc)) # This is d on the paper
degreesPerSubdivision = (360/initialProjections) / subdivisions

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

## Function that selects roughly the number numelems as a sample. (You get substantially less)
## Say you input numelems=90. Then you get less than 90 voxels in your case.
def get_sub_sub_sample(subsampling_img, numelems):
    sub_sub = np.zeros_like(subsampling_img)
    locations = np.where(subsampling_img)[0]
    print(locations)
    print('number of elements', len(locations))
    a = np.arange(0,len(locations), int(len(locations)/numelems))
    print(a)
    sublocations = locations[a]
    sub_sub[sublocations] = 1
    return(sub_sub)

class tomodata:
    ## Initialization of the data
    def __init__(self):
        self.base_dir = 'data/dij/HelicalGyn/'
        #self.base_dir = 'data/dij153/prostate/'#153
        self.base_dir = 'data/dij/prostate/'  # 51
        # The number of loops to be used in this case
        self.ProjectionsPerLoop = initialProjections
        self.bixelsintween = 5
        self.maxIntensity = 300
        self.yBar = 350
        self.maxvoxels = maxvoxels
        self.tprime = timeko # LOT in miliseconds
        self.tdprime = timekc # LCT in miliseconds
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
        self.argumentVariables()
        print('Read vectors...')
        self.readWeiguosCase(  )
        self.maskNamesGetter(self.base_dir + self.struct_img_header)
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        # Now remove bixels carefully
        # self.removebixels(self.bixelsintween)
        # Do the smallvoxels again:
        blaa, blab, self.smallvoxels, blad = np.unique(self.smallvoxels, return_index=True, return_inverse=True, return_counts=True)
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
        for i in range(len(self.mask)):
            # Constraint on TARGETS
            T = None
            if self.mask[i] in self.TARGETList:
                T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                self.quadHelperOver[i] = 0.0002
                self.quadHelperUnder[i] = 0.08
            # Constraint on OARs
            elif self.mask[i] in self.OARList:
                T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                self.quadHelperOver[i] = 0.000002
                self.quadHelperUnder[i] = 0.0
            elif 0 == self.mask[i]:
                print('there is an element in the voxels that is also mask 0')
            self.quadHelperThresh[i] = T
        self.subdivisionOrganizer()
            ########################

    def subdivisionOrganizer(self):
        self.numProjections *= subdivisions
        self.ProjectionsPerLoop *= subdivisions

    def argumentVariables(self):
        if len(sys.argv) > 1:
            self.bixelsintween = int(sys.argv[1])
        if len(sys.argv) > 2:
            self.maxvoxels = int(sys.argv[2])
        if len(sys.argv) > 3:
            self.MBar = int(sys.argv[3])

    ## Keep the ROI's in a dictionary
    def maskNamesGetter(self, maskfile):
        lines = tuple(open(maskfile, 'r'))
        for line in lines:
            if 'ROIIndex =' == line[:10]:
                roinumber = line.split(' = ')[1].strip()
            elif 'ROIName =' == line[:9]:
                roiname = line.split(' = ')[1].strip()
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
                if 'ROIIndex =' in line:
                    roiidx = int(line.split(' ')[2])
                elif 'ROIName =' in line:
                    roiname = line.split(' ')[2]
                elif 'RTROIInterpretedType =' in line:
                    roitype = line.split(' ')[2]
                    if 'SUPPORT' in roitype:
                        self.SUPPORTDict[roiidx] = roiname
                    elif 'ORGAN' in roitype:
                        self.OARDict[roiidx] = roiname
                    elif 'PTV' in roitype:
                        self.TARGETDict[roiidx] = roiname
                    else:
                        sys.exit('ERROR, roi type not defined')
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
        print(len(locats))
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

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        # Assign structures and thresholds for each of them
        self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22]
        self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 78, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.TARGETList = [2]
        self.TARGETThresholds = [78]
        dtype=np.uint32

        self.bixels = getvector(self.base_dir + 'dij/Bixels_out.bin', np.int32)
        self.voxels = getvector(self.base_dir + 'dij/Voxels_out.bin', np.int32)
        self.Dijs = getvector(self.base_dir + 'dij/Dijs_out.bin', np.float32)
        #self.voxelsshort = getvector('data/dij/prostate/' + 'dij/Voxels_out.bin', np.int32)
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
        self.removezeroes([0, 10, 11, 17, 12, 3, 15, 16, 9, 5, 4, 20, 21, 19, 18])

def solveContinuous (data):
    voxels = range(len(data.mask))
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = projIni * subdivisions
    projections = range(numProjections)
    leaves = range(data.L)
    leafsD = (data.bixels % data.L).astype(int)
    projectionsD = np.floor(data.bixels / data.L).astype(int)
    m = Model("SOLVECONTINUOUS")
    m.params.BarConvTol = 1.0
    print("Solving the continuous version of the model")
    z = m.addVars(voxels, lb = 0.0, obj = 1.0, vtype = GRB.CONTINUOUS, names = "z")
    z_plus = m.addVars(voxels, lb = 0.0, obj = np.sqrt(data.quadHelperOver), vtype = GRB.CONTINUOUS, names = "z_plus")
    z_minus = m.addVars(voxels, lb = 0.0, obj = np.sqrt(data.quadHelperUnder), vtype = GRB.CONTINUOUS, names = "z_minus")
    dose = m.addVars(leaves, projections, obj=1.0, vtype=GRB.CONTINUOUS, names="dose", lb = timeko, ub = t51 - timekc)
    myObj = QuadExpr(0.0)
    hs = [LinExpr(0.0) for _ in voxels]
    [hs[data.smallvoxels[l]].add(data.Dijs[l] * dose[leafsD[l], projectionsD[l]]) for l in range(len(data.smallvoxels))]
    [m.addConstr(z[v] == data.yBar * hs[v], name="doses_to_j_yparam[" + str(v) + "]") for v in voxels]
    for v in voxels:
        myObj.add(z_minus[v] * z_minus[v] + z_plus[v] * z_plus[v])
    positive_only = m.addConstrs((z_plus[v] - z_minus[v] == z[v] - data.quadHelperThresh[v] for v in voxels), "positive_only")
    m.setObjective(myObj, GRB.MINIMIZE)
    m.update()
    m.optimize()
    z_output = [v.x for v in m.getVars()[0:len(voxels)]]
    d = {"z_out": z, "z_plus_out": z_plus, "z_minus_out": z_minus, "dose_out": dose}
    return(d)

def preTreatmentMaterials(data, d):
    # How many projections must remain open per leaf
    ko = math.ceil(howmanydegreesko / degreesPerSubdivision)
    kc = math.ceil(howmanydegreeskc / degreesPerSubdivision)
    k10 = math.ceil(howmanydegrees10 / degreesPerSubdivision)
    kcomax = max(ko, kc)
    projIni = int(1 + np.floor(max(data.bixels / data.L)).astype(int))
    projBasic = projIni * subdivisions
    numProjections = k10 + projBasic + (kcomax - 1)
    Pset = [numProjections for _ in range(data.L)]
    Psetshort = [numProjections - 1 for _ in range(data.L)]
    PsetshortM1 = [numProjections - 2 for _ in range(data.L)]
    M = [np.concatenate((np.concatenate(([-10**3] * k10, np.repeat(range(projIni), subdivisions)), axis=0), [-10**3]*(kcomax - 1)), axis=0) for _ in range(data.L)]
    Minv = [defaultdict(list) for _ in range(data.L)] # Minv is a list of defaultdicts that keeps the inverse of map M
    for l in range(len(M)):
        for p in range(len(M[l])):
            Minv[l][M[l][p]].append(p)
    deltalp = delta51 / subdivisions
    deltas = [[deltalp for _ in range(Pset[l])] for l in range(data.L)]
    LOTset = [[range(ko) if p <= (Pset[l] - kcomax) else range(0) for p in range(Pset[l])] for l in range(data.L)] # This range must be reviewed
    LCTset = [[range(kc) if p <= (Pset[l] - kcomax) else range(0) for p in range(Pset[l])] for l in range(data.L)]
    # Now let's recalculate the z's and accomodate them to the binary world.
    betas_in = []
    B_in = []
    cgamma_in = []
    lgamma_in = []
    for l in range(data.L):
        betas_in.append([0] * Pset[l])
        for p in range(Pset[l]):
            betas[l][p]

    d = {"Pset": Pset, "Psetshort": Psetshort, "PsetshortM1": PsetshortM1, "M": M, "Minv": Minv, "deltas": deltas, "LOTset": LOTset, "LCTset": LCTset, "k10": k10,
         "z_out": d["z_out"], "z_plus_out": d["z_plus_out"], "z_minus_out": d["z_minus_out"], "dose_out": d["dose_out"]}
    return(d)

def warmTreatment(data, d):
    # How many projections must remain open per leaf
    ko = math.ceil(howmanydegreesko / degreesPerSubdivision)
    kc = math.ceil(howmanydegreeskc / degreesPerSubdivision)
    k10 = math.ceil(howmanydegrees10 / degreesPerSubdivision)
    kcomax = max(ko, kc)
    Pset = []
    M = []
    for l in range(data.L):
        Pset.append([i for i in range(k10)])
        M.append([-10**3] * k10)
        deltas.append([delta51] * k10)
        i = k10
        for p in range(k10, d["Pset"][l] - (kcomax - 1)):
            if 1 == betas[l][p]:
                segments = subdivisions
            else:
                segments = 1
            for _ in segments:
                M[l].append(d["M"][l][p])
                deltas[l].append(d["deltas"][l][p] / segments)
                Pset[l].append(i)
                i += 1
        for p in kcomax - 1:
            M[l].append(-10**3)
            Pset[l].append(i)
            deltas[l].append(delta51)
            i += 1
    Psetshort = [Pset[l].pop() for _ in range(data.L)]
    PsetshortM1 = [Psetshort[l].pop() - 2 for _ in range(data.L)]
    Minv = [defaultdict(list) for _ in range(data.L)] # Minv is a list of defaultdicts that keeps the inverse of map M
    LOTset = []
    LCTset = []
    for l in range(len(M)):
        LOTset.append(np.array([0.0] * len(M[l])))
        LCTset.append(np.array([0.0] * len(M[l])))
        # fulltimes is how much of the time is getting covered by following
        ofulltimes = np.array([0.0] * len(M[l]))
        cfulltimes = np.array([0.0] * len(M[l]))
        c0 = 0
        o0 = 0
        for p in range(len(M[l])):
            Minv[l][M[l][p]].append(p)
            ofulltimes[o0:p] += deltas[l][p]
            cfulltimes[c0:p] += deltas[l][p]
            LOTset[l][o0:p] += 1
            LCTset[l][c0:p] += 1
            while(ofulltimes[o0] >= howmanydegreesko):
                o0 += 1
            while(cfulltimes[c0] >= howmanydegreeskc):
                c0 += 1

    dnew = {"Pset": Pset, "Psetshort": Psetshort, "PsetshortM1": PsetshortM1, "M": M, "Minv": Minv, "deltas": deltas,
            "LOTset": LOTset, "LCTset": LCTset, "k10": k10, "z_out": d["z_out"], "z_plus_out": d["z_plus_out"],
            "z_minus_out": d["z_minus_out"], "betas_out": d["betas_out"], "B_out":d["B_out"], "cgamma_out": d["cgamma_out"],
            "lgamma_out": d["lgamma_out"]}
    return(dnew)

def createModel (data, d):
    voxels = range(len(data.mask))
    leaves = range(data.L)
    leafsD = (data.bixels % data.L).astype(int)
    projectionsD = np.floor(data.bixels / data.L).astype(int)
    projIni = 1 + np.floor(max(data.bixels / data.L)).astype(int)
    numProjections = projIni * subdivisions
    projections = range(numProjections)
    projectionsshort = range(numProjections - 1)
    projectionsshortM1 = range(numProjections - 2)
    flag = True
    for _ in range(2):
        if flag:
            d = preTreatmentMaterials(data, d)
        else:
            d = warmTreatment(data, d)
        m = Model("SOLVE51")
        m.params.MIPGap = 2E-1
        m.params.IntFeasTol = 1E-2
        print('Solving the MIP')
        z = m.addVars(voxels, lb = 0.0, obj = 1.0, vtype = GRB.CONTINUOUS, names = "z")
        z_plus = m.addVars(voxels, lb = 0.0, obj = np.sqrt(data.quadHelperOver), vtype = GRB.CONTINUOUS, names = "z_plus")
        z_minus = m.addVars(voxels, lb = 0.0, obj = np.sqrt(data.quadHelperUnder), vtype = GRB.CONTINUOUS, names = "z_minus")
        betas = [m.addVars(range(d["Pset"][l]), obj = 1.0, vtype = GRB.BINARY, names = "betas_" + str(l)) for l in range(data.L)]
        B = [m.addVars(range(d["Pset"][l]), obj = 1.0, vtype = GRB.BINARY, names = "B_" + str(l)) for l in range(data.L)]
        cgamma = [m.addVars(range(d["Pset"][l]), obj = 1.0, vtype = GRB.BINARY, names = "cgamma_" + str(l)) for l in range(data.L)]
        lgamma = [m.addVars(range(d["Pset"][l]), obj = 1.0, vtype = GRB.BINARY, names = "lgamma_" + str(l)) for l in range(data.L)]
        m.update()
        # Create Constraints
        positive_only = m.addConstrs((z_plus[v] - z_minus[v] == z[v] - data.quadHelperThresh[v] for v in voxels), "positive_only")
        doses_to_j_yparam = []
        myObj = QuadExpr(0.0)
        rhs = [LinExpr(0.0) for _ in voxels]
        [rhs[data.smallvoxels[i]].add(data.Dijs[i] * quicksum([betas[leafsD[i]][p_hat] for p_hat in d["Minv"][leafsD[i]][projectionsD[i]]])) for i in range(len(data.smallvoxels))]
        for v in voxels:
            doses_to_j_yparam.append(m.addConstr(z[v] == data.yBar * rhs[v], name="doses_to_j_yparam[" + str(v) + "]"))
            # Append goal to objective
            # DON'T MULTIPLY TIMES THE QUADHELPERS! THIS IS ALREADY DONE IN THE DECLARATION 10 LINES ABOVE
            myObj.add(z_minus[v] * z_minus[v] + z_plus[v] * z_plus[v])
        LOC = m.addConstrs((B[l][p] <= betas[l][p + k] for l in leaves for p in range(d["Pset"][l]) for k in d["LOTset"][l][p]), "LOC")
        LCT = m.addConstrs((cgamma[l][p] <= lgamma[l][p + k] for l in leaves for p in range(d["Pset"][l]) for k in d["LCTset"][l][p]), "LCT")
        endOpen = m.addConstrs((betas[l][p] <= betas[l][p + 1] + cgamma[l][p + 1] for l in leaves for p in range(d["Psetshort"][l])), "endOpen")
        endClose = m.addConstrs((lgamma[l][p] <= B[l][p + 1] + lgamma[l][p + 1] for l in leaves for p in range(d["Psetshort"][l])), "endClose")
        eitherOpenOrClose = m.addConstrs((betas[l][p] + lgamma[l][p] == 1 for l in leaves for p in range(d["Pset"][l])), "eitherOpenOrClose")
        m.setObjective(myObj, GRB.MINIMIZE)
        m.update()
        if flag:
            for v in voxels:
                a = 1
                #z[v].Pstart = d["z_out"][v]
                #z_plus[v].Pstart = d["z_plus_out"][v]
                #z_minus[v].Pstart = d["z_minus_out"][v]
        else:
            for v in voxels:
                z[v].Start = d["z_out"][v]
                z_plus[v].Start = d["z_plus_out"][v]
                z_minus[v].Start = d["z_minus_out"][v]
            for l in range(len(d["M"])):
                for p in range(d["Pset"][l]):
                    betas[l][p].Start = d["betas_out"][l][p]
                    B[l][p].Start = d["B_out"][l][p]
                    cgamma[l][p].Start = d["cgamma_out"][l][p]
                    lgamma[l][p].Start = d["lgamma_out"][l][p]
        m.optimize()
        z_output = [v.x for v in m.getVars()[0:len(voxels)]]
        d["z_out"] = z
        d["z_plus_out"] = z_plus
        d["z_minus_out"] = z_minus
        d["betas_out"] = betas
        d["B_out"] = B
        d["cgamma_out"] = cgamma
        d["lgamma_out"] = lgamma
        flag = False
    return(z_output)

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
    plt.title('DVH')
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.savefig(data.outputDirectory + NameTag + '.png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    if showPlot:
        plt.show()
    plt.close()

dataobject = tomodata()
d = solveContinuous(dataobject)
z = createModel(dataobject, d)
plotDVHNoClass(dataobject, d["z_out"], 'dvh')
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


#    for l in range(data.L):
#        print('adding leaf:', l)
#        for p in range(d["Pset"][l]):
#            loc = np.where(d["M"][l][p] * data.L + l == data.bixels)[0]
#            Dijs = [0.0 for _ in voxels]
#            for i in loc:
#                Dijs[data.smallvoxels[i]] += data.Dijs[i]
#            for v in voxels:
#                rhs[v] += Dijs[v] * d["deltas"][l][p] * betas[l][p]