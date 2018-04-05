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
import os.path
from scipy.stats import describe
import subprocess
import sys
import pickle
import math
import time

numcores = 8
subdivisions = 1
maxvoxels = 200
tumorsite = "Prostate"
degreesPerSubdivision = (360/51) / subdivisions
timeko = 0.05 # msecs
timekc = 0.04 # msecs
speed = 24 # degrees per second
howmanydegreesko = speed * timeko # Degrees spanned in this time
howmanydegreeskc = speed * timekc # Degrees spanned in this time
ko = math.ceil(howmanydegreesko / degreesPerSubdivision)
kc = math.ceil(howmanydegreeskc / degreesPerSubdivision)

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
        self.ProjectionsPerLoop = 51
        self.bixelsintween = 5
        self.maxIntensity = 300
        self.yBar = 350
        self.maxvoxels = maxvoxels
        self.tprime = 100 # LOT in miliseconds
        self.tdprime = 40 # LCT in miliseconds
        self.img_filename = 'samplemask.img'
        self.header_filename = 'samplemask.header'
        self.struct_img_filename = 'roimask.img'
        self.struct_img_header = 'roimask.header'
        self.outputDirectory = "output/"
        self.roinames = {}
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.N = 80
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
                self.quadHelperOver[i] = 0.00002
                self.quadHelperUnder[i] = 0.00008
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
        # Convert the mask into a list of unitary structures. A voxel gets assigned to only one place.
        img_struct = get_structure_mask(reversed(self.ALLList), struct_img_arr)
        # Get the subsampled list of voxels.
        self.mask = get_subsampled_mask(img_struct, img_arr)
        # Select only the voxels that exist in the small voxel space provided.
        self.removezeroes([0, 10, 11, 17, 12, 3, 15, 16, 9, 5, 4, 20, 21, 19, 18])

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
## This part is for AMPL's implementation:
def printAMPLfile(data, subdivisions, tumorsite):
    f = open("Data" + str(tumorsite) + "/tomononlinearRealCases_split_" + str(subdivisions) + "_vxls_" + str(data.maxvoxels) + ".dat", "w")
    print('param numProjections :=', data.numProjections, ';', file = f)
    print('param U :=', data.maxIntensity, ';', file = f)
    print('param numLeaves :=', data.N, ';', file=f)
    print('param yparam :=', data.yBar, ';', file=f)
    print('param projecs :=', data.ProjectionsPerLoop, ';', file=f)
    print('param numvoxels :=', len(data.mask), ';', file=f)
    print('param kc :=', kc, ';', file=f)
    print('param ko :=', ko, ';', file=f)
    pds.set_option('precision', 16)
    leafs = (data.bixels % data.N).astype(int)
    projections = np.floor(data.bixels / data.N).astype(int)

    # Incorporate subdivisions
    leafs = np.repeat(leafs, subdivisions)
    data.smallvoxels = np.repeat(data.smallvoxels, subdivisions)
    projections = subdivisions * np.repeat(projections, subdivisions) + np.tile([i for i in range(subdivisions)], len(data.Dijs))
    # THIS HAS TO BE DONE AFTER THE PREVIOUS STEP
    data.Dijs = np.repeat(data.Dijs, subdivisions) / subdivisions
    # Done with the incorporation of subdivisions
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
    print('param: KNJMPARAMETERS: D:=' , file = f)
    sparseinfo = pds.DataFrame(data = {'LEAVES' : leafs, 'PROJECTIONS' : projections, 'VOXELS' : data.smallvoxels, 'ZDOSES' : data.Dijs},
                               columns = ['LEAVES', 'PROJECTIONS', 'VOXELS', 'ZDOSES'])
    print(sparseinfo.to_string(index=False, header=False), file = f)
    print(";", file = f)
    f.close()

def runAMPL(maxvoxels, i, tumorsite):
    # Change the run file:
    with open("LOTModelTemplate.run", "r+") as r:
        with open("LOTModel.run", "w") as w:
            old = r.read()
            newstatement = old
            if old.startswith("data "):
                newstatement = "data Data" + tumorsite + "/tomononlinearRealCases_split_" + str(i) + "_vxls_" + str(maxvoxels) + ".dat;"
            w.write(newstatement)
    procstring = subprocess.check_output(['ampl', 'LOTModel.run'])
    return(procstring)

def writebetas(betas, B, cgamma, lgamma, numProjections, maxkcko, numLeaves):
    f = open("warmstart.dat", "w")
    leaves = np.tile(np.array(range(numLeaves)), numProjections + 2 * maxkcko)
    projections = np.repeat(np.concatenate((np.concatenate((np.array(range(-maxkcko, 0)), np.array(range(numProjections)))), np.array(range(numProjections, numProjections + maxkcko)))), numLeaves)
    print("param betas := ", file=f)
    betainfo = pds.DataFrame(
        data={'LEAVES': leaves, 'PROJECTIONS': projections, 'betas': betas.flatten()},
        columns=['LEAVES', 'PROJECTIONS', 'betas'])
    print(betainfo.to_string(index=False, header=False), file=f)
    print(";", file=f)
    print("param B := ", file=f)
    betainfo = pds.DataFrame(
        data={'LEAVES': leaves, 'PROJECTIONS': projections, 'B': B.flatten()},
        columns=['LEAVES', 'PROJECTIONS', 'B'])
    print(betainfo.to_string(index=False, header=False), file=f)
    print(";", file=f)
    print("param cgamma := ", file=f)
    betainfo = pds.DataFrame(
        data={'LEAVES': leaves, 'PROJECTIONS': projections, 'cgamma': cgamma.flatten()},
        columns=['LEAVES', 'PROJECTIONS', 'cgamma'])
    print(betainfo.to_string(index=False, header=False), file=f)
    print(";", file=f)
    print("param lgamma := ", file=f)
    betainfo = pds.DataFrame(
        data={'LEAVES': leaves, 'PROJECTIONS': projections, 'lgamma': lgamma.flatten()},
        columns=['LEAVES', 'PROJECTIONS', 'lgamma'])
    print(betainfo.to_string(index=False, header=False), file=f)
    print(";", file=f)
    f.close()

def readDosefromtext(pstring):
    strstring = pstring.decode("utf-8") # decode the bytes stringst
    #print(strstring)
    lines = strstring.split('\n')
    linecontainssolution = False
    linecontainsbetas = False
    linecontainsB = False
    linecontainslgamma = False
    linecontainscgamma = False
    obj = None
    for line in lines:
        if line.startswith("numvoxels"):
            numvoxels = line.split()[2]
            z = np.zeros(int(numvoxels))
        elif line.startswith("ObjectiveFunction"):
            obj = float(line.split()[2])
            print('objective value in the latest run:', obj)
        elif line.startswith("maxkcko"):
            maxkcko = int(line.split()[2])
        elif line.startswith('numProjections'):
            numProjections = int(line.split()[2])
        elif line.startswith('numLeaves'):
            numLeaves = int(line.split()[2])
            betas = np.zeros((numProjections + 2 * maxkcko, numLeaves))
            B = np.zeros((numProjections + 2 * maxkcko, numLeaves))
            cgamma = np.zeros((numProjections + 2 * maxkcko, numLeaves))
            lgamma = np.zeros((numProjections + 2 * maxkcko, numLeaves))
        elif linecontainssolution:
            l = []
            for t in line.split():
                try:
                    l.append(float(t))
                except ValueError:
                    pass
            if len(l) > 0:
                for i in range(int(len(l) / 2)):
                    z[int(l[int(2 * i)])] = l[int(2 * i + 1)]
            if "" == line:
                linecontainssolution = False
        elif linecontainsbetas:
            if not line == "": #If it's not empty
                if line.startswith(":"):
                    locat = []
                    for t in line.split():
                        try:
                            locat.append(int(t))
                        except ValueError:
                            pass
                else:
                    l = []
                    for t in line.split():
                        try:
                            l.append(int(t))
                        except ValueError:
                            pass
                    for i in range(int(len(l) - 1)):
                        betas[l[0] + maxkcko, locat[i]] = l[i + 1]
        elif linecontainsB:
            if not line == "": #If it's not empty
                if line.startswith(":"):
                    locat = []
                    for t in line.split():
                        try:
                            locat.append(int(t))
                        except ValueError:
                            pass
                else:
                    l = []
                    for t in line.split():
                        try:
                            l.append(int(t))
                        except ValueError:
                            pass
                    for i in range(int(len(l) - 1)):
                        B[l[0] + maxkcko, locat[i]] = l[i + 1]
        elif linecontainscgamma:
            if not line == "": #If it's not empty
                if line.startswith(":"):
                    locat = []
                    for t in line.split():
                        try:
                            locat.append(int(t))
                        except ValueError:
                            pass
                else:
                    l = []
                    for t in line.split():
                        try:
                            l.append(int(t))
                        except ValueError:
                            pass
                    for i in range(int(len(l) - 1)):
                        cgamma[l[0] + maxkcko, locat[i]] = l[i + 1]
        elif linecontainslgamma:
            if not line == "": #If it's not empty
                if line.startswith(":"):
                    locat = []
                    for t in line.split():
                        try:
                            locat.append(int(t))
                        except ValueError:
                            pass
                else:
                    l = []
                    for t in line.split():
                        try:
                            l.append(int(t))
                        except ValueError:
                            pass
                    for i in range(int(len(l) - 1)):
                        lgamma[l[0] + maxkcko, locat[i]] = l[i + 1]
        elif ('z [*] :=' in line):
            linecontainssolution = True
        elif ('betas [*,*] (tr)' in line):
            linecontainsbetas = True
        elif ('B [*,*] (tr)' in line):
            linecontainsbetas = False
            linecontainsB = True
        elif ('cgamma [*,*] (tr)' in line):
            linecontainsB = False
            linecontainscgamma = True
        elif ('lgamma [*,*] (tr)' in line):
            linecontainscgamma = False
            linecontainslgamma = True
    writebetas(betas, B, cgamma, lgamma, numProjections, maxkcko, numLeaves)
    return(z, betas, B, cgamma, lgamma, obj)

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

# Only enter the next loop if the file exists
if not os.path.isfile("Data" + str(tumorsite) + "/tomononlinearRealCases_split_" + str(subdivisions) + "_vxls_" + str(maxvoxels) + ".dat"):
    dataobject = tomodata()
    printAMPLfile(dataobject, subdivisions, tumorsite)

start_time = time.time()
# Erase everything in the warmstart file
f = open("warmstart.dat", "w")
f.close()
oldobj = np.inf
for i in [1, 2, 4, 8, 16, 32]:
    start_time = time.time()
    pstring = runAMPL(maxvoxels, i, tumorsite)
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
