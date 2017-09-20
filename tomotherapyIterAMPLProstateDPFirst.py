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
import scipy.sparse as sps
import time
from scipy.stats import describe
import subprocess

numcores = 4

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
# In[5]:

def get_structure_mask(struct_id_list, struct_img_arr):
    img_struct = np.zeros_like(struct_img_arr)
    for s in struct_id_list:
        img_struct[np.where(struct_img_arr & 2 ** (s - 1))] = s
    return np.copy(img_struct)

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
        self.base_dir = 'data/dij/prostate/'
        self.img_filename = 'samplemask.img'
        self.header_filename = 'samplemask.header'
        self.struct_img_filename = 'roimask.img'
        self.struct_img_header = 'roimask.header'
        self.outputDirectory = "output/"
        # Are we going to sample?
        self.sampleevery = 32
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.N = 80
        self.maxIntensity = 500
        self.caseSideX = 256  ## Wilmer what is going on here.
        self.caseSideY = 256
        self.caseSideZ = 193
        # The number of loops to be used in this case
        self.numloops = 8.27
        self.voxelsBigSpace = self.caseSideX * self.caseSideY * self.caseSideZ
        print('Read vectors...')
        self.readWeiguosCase(  )
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        print('Build sparse matrix.')
        # The next part uses the case corresponding to either Wilmer or Weiguo's case
        self.totalbeamlets = 33792
        self.totalsmallvoxels = max(self.smallvoxels) + 1 #12648448
        print('totalsmallvoxels:', self.totalsmallvoxels)
        print('a brief description of Dijs array', describe(self.Dijs))
        self.D = sps.csr_matrix((self.Dijs, (self.smallvoxels, self.bixels)), shape=(self.totalsmallvoxels, self.totalbeamlets))
        self.quadHelperThresh = np.zeros(len(self.mask))
        self.quadHelperUnder = np.zeros(len(self.mask))
        self.quadHelperOver = np.zeros(len(self.mask))
        self.numProjections = self.getNumProjections()
        #######################################3
        for i in range(len(self.mask)):
            # Constraint on TARGETS
            T = None
            if self.mask[i] in self.TARGETList:
                T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                self.quadHelperOver[i] = 0.0001
                self.quadHelperUnder[i] = 0.006
            # Constraint on OARs
            elif self.mask[i] in self.OARList:
                T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                self.quadHelperOver[i] = 0.0001
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

    def getNumProjections(self):
        with open(self.base_dir + 'motion.txt') as f:
            for i, l in enumerate(f):
                pass
        return i # Do not return -1 because the file has a header.

    def removezeroes(self):
        # Next I am removing the voxels that have a mask of zero (0) because they REALLY complicate things otherwise
        # Making the problem larger.
        #-------------------------------------
        # Cut the mask to only the elements contained in the voxel list
        voxelindex = np.zeros_like(self.mask)
        voxelindex[np.unique(self.voxels)] = 1
        self.mask = np.multiply(voxelindex, self.mask)
        locats = np.where(0 == self.mask)[0]
        self.mask = np.delete(self.mask, locats)
        # intersection of voxels and nonzero
        indices = np.where(np.in1d(self.voxels, locats))[0]
        # Cut whatever is not in the voxels.
        self.bixels = np.delete(self.bixels, indices)
        self.voxels = np.delete(self.voxels, indices)
        self.Dijs = np.delete(self.Dijs, indices)
        #self.mask = np.delete(self.mask, indices)

    def removebixels(self, pitch):
        bixelkill = np.where(0 != (self.bixels % pitch) )
        self.bixels = np.delete(self.bixels, bixelkill)
        self.voxels = np.delete(self.voxels, bixelkill)
        self.Dijs = np.delete(self.Dijs, bixelkill)


    def convertmasktobasic(self):
        ## Get only the basic bit from the mask.
        inorgan = [100] * (len(self.mask))
        for i in reversed(self.ALLList):
            print(i)
            inorgan = [i if self.mask[n] & 2**(i-1) else inorgan[n] for n in range(len(self.mask))]
        self.mask = inorgan
        np.save('data/dij/prostate/dij/roimaskClean', arr=self.mask, allow_pickle=False)

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        if self.base_dir=='data/dij/prostate/':
            dtype=np.uint32
            # Assign structures and thresholds for each of them
            self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22]
            self.OARThresholds = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            self.TARGETList = [2]
            self.TARGETThresholds = [78]
        else:
            dtype=np.uint16
            self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22]
            self.OARThresholds = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            self.TARGETList = [2]
            self.TARGETThresholds = [78]

        self.bixels = getvector(self.base_dir + 'dij/Bixels_out.bin', np.int32)
        self.voxels = getvector(self.base_dir + 'dij/Voxels_out.bin', np.int32)
        self.Dijs = getvector(self.base_dir + 'dij/Dijs_out.bin', np.float32)

        self.ALLList = self.TARGETList + self.OARList

        # get subsample mask
        img_arr = getvector(self.base_dir + self.img_filename, dtype=dtype)
        img_arr = get_sub_sub_sample(img_arr, 90)
        # get structure file
        struct_img_arr = getvector(self.base_dir + self.struct_img_filename, dtype=dtype)
        # Convert the mask into a list of unitary structures. A voxel gets assigned to only one place.
        img_struct = get_structure_mask(reversed(self.ALLList), struct_img_arr)
        # Get the subsampled list of voxels.
        self.mask = get_subsampled_mask(img_struct, img_arr)
        # Select only the voxels that exist in the small voxel space provided.
        self.removezeroes()
        #self.convertmasktobasic()
        self.removebixels(5)

## Number of beamlets in each gantry. Usually 64 but Weiguo uses 80
## This part is for AMPL's implementation:
def printAMPLfile(data):
    f = open("tomononlinearRealCases.dat", "w")
    print('param numProjections :=', data.numProjections, ';', file = f)
    print('param numvoxels :=', len(data.mask), ';', file = f)
    print('param U :=', data.maxIntensity, ';', file = f)
    print('param numLeaves :=', data.N, ';', file=f)
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
    pds.set_option('precision', 16)
    leafs = (data.bixels % data.N).astype(int)
    projections = np.floor(data.bixels / data.N).astype(int)
    sparseinfo = pds.DataFrame(data = {'LEAVES' : leafs, 'PROJECTIONS' : projections, 'VOXELS' : data.smallvoxels, 'ZDOSES' : data.Dijs})
    print(sparseinfo.to_string(index=False, header=False), file = f)
    print(";", file = f)
    f.close()

def runAMPL():
    procstring = subprocess.check_output(['ampl', 'heuristicRealCasesDPfirst.run'])
    return(procstring)

def readDosefromtext(pstring):
    strstring = pstring.decode("utf-8") # decode the bytes stringst
    print(strstring)
    lines = strstring.split('\n')
    linecontainssolution = False
    for line in lines:
        if linecontainssolution:
            l = []
            for t in line.split():
                try:
                    l.append(float(t))
                except ValueError:
                    pass
            if len(l) > 0:
                for i in range(int(len(l) / 2)):
                    z[int(l[int(2 * i)])] = l[int(2 * i + 1)]
        else:
            if ('z [*] :=' in line):
                linecontainssolution = True
    return(z)

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
        plt.plot(bins, dvh, label="struct " + str(index), linewidth=2)
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

start_time = time.time()
dataobject = tomodata()
printAMPLfile(dataobject)
z = np.zeros(len(dataobject.mask))
pstring = runAMPL()
z = readDosefromtext(pstring)
plotDVHNoClass(dataobject, z, 'dvh')
# Output ampl results for the next run in case something fails.
text_output = open("amploutput.txt", "wb")
text_output.write(pstring)
text_output.close()
print("--- %s seconds ---" % (time.time() - start_time))
