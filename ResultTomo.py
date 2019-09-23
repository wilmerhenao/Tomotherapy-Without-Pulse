
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

outputDirectory = "outputMultiProj/"
imrt = False
imrtwith20msecondsconstraint = False
showPlot = False


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
        if tumorsite == "Prostate153":
            self.base_dir = 'data/dij153/prostate/'  # 153
            if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
                self.base_dir = '/scratch/engin_flux/wilmer/dij/prostate/'
        if tumorsite == "Lung":
            self.base_dir = 'data/dij153/lung/'  # 153
            if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):  # FLUX
                self.base_dir = '/scratch/engin_flux/wilmer/dij/lung/'
        # The number of loops to be used in this case
        self.ProjectionsPerLoop = initialProjections
        self.bixelsintween = 1
        self.yBar = 39 * 9.33
        self.yBar = 700
        self.maxvoxels = maxvoxels
        self.img_filename = 'samplemask.img'
        self.header_filename = 'samplemask.header'
        self.struct_img_filename = 'roimask.img'
        self.struct_img_header = 'roimask.header'
        self.outputDirectory = "outputMultiProj/"
        self.roinames = {}
        # N Value: Number of beamlets in the gantry (overriden in Wilmer's Case)
        self.L = numberOfLeaves
        self.get_dim(self.base_dir, 'samplemask.header')
        self.get_totalbeamlets(self.base_dir, 'dij/Size_out.txt')
        self.roimask_reader(self.base_dir, 'roimask.header')
        self.timeA = timeA
        self.timeM = timeM
        #self.argumentVariables()
        print('Read vectors...')
        self.readWeiguosCase(  )
        self.maskNamesGetter(self.base_dir + self.struct_img_header)
        print('done')
        # Create a space in smallvoxel coordinates
        self.smallvoxels = self.BigToSmallCreator()
        #Now remove bixels carefully
        #Do the smallvoxels again:
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
        projIni = 1 + np.floor(max(self.bixels / self.L)).astype(int)
        self.numProjections = k10 + projIni
        self.leafsD = (self.bixels % self.L).astype(int)
        self.projectionsD = np.floor(self.bixels / self.L).astype(int)
        self.bdata = np.zeros((self.numProjections, self.L))
        if tumorsite == "Prostate":
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 15.5 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 0.023 #PROSTATE! FOR GYN SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 5E-3 #PROSTATE! FOR GYN SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #PROSTATE! FOR GYN SEE BELOW!
                    #if 7 == self.mask[i]: # RECTUM!!!
                    #    self.quadHelperOver[i] = 50E-3  # PROSTATE! FOR GYN SEE BELOW!
                    #    self.quadHelperUnder[i] = 0.0  # PROSTATE! FOR GYN SEE BELOW!
                elif 0 == self.mask[i]:
                    print('there is an element in the voxels that is also mask 0')
                self.quadHelperThresh[i] = T
                ########################
        elif "Lung"  == tumorsite:
            print('Gyn Case parameters')
            for i in range(len(self.mask)):
                # Constraint on TARGETS
                T = None
                if self.mask[i] in self.TARGETList:
                    T = self.TARGETThresholds[np.where(self.mask[i] == self.TARGETList)[0][0]]
                    self.quadHelperOver[i] = 0.0001 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 1E1 #GYN! FOR PROSTATE SEE BELOW!
                # Constraint on OARs
                elif self.mask[i] in self.OARList:
                    T = self.OARThresholds[np.where(self.mask[i] == self.OARList)[0][0]]
                    self.quadHelperOver[i] = 0.003 #GYN! FOR PROSTATE SEE BELOW!
                    self.quadHelperUnder[i] = 0.0 #GYN! FOR PROSTATE SEE BELOW!
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
        for i in range(len(self.bixels)):
            beamP = self.projectionsD[i] + k10
            beamL = self.leafsD[i]
            if self.bdata[beamP, beamL] < self.Dijs[i]:
                self.bdata[beamP, beamL] = self.Dijs[i]
        # Logging
        self.treatmentName = 'IMRT'
        if imrtwith20msecondsconstraint:
            self.treatmentName += '20msec'
        if not imrt:
            if pairSolution:
                self.treatmentName = 'pairModel'
            else:
                self.treatmentName = 'fullModel'
            if relaxedProblem:
                self.treatmentName += 'relaxedVersion'
        self.chunkName = tumorsite + '-' + str(initialProjections) + '-' + self.treatmentName + '-MinLOT-' + str(timeM) + '-minAvgLot-' + str(timeA) + '-vxls-' + str(self.totalsmallvoxels) + '-ntnsty-' + str(self.yBar)
        self.logfile = ''
        if imrt:
            self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'IMRT.log'
        else:
            if relaxedProblem:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'relaxed.log'
            elif pairSolution:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'pairSolution.log'
            else:
                self.logFile = self.outputDirectory + 'logFile' + self.chunkName + 'completeSolution.log'

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
                    elif 'PTV' in roitype or 'CTV' in roitype or 'TARGET' in roitype:
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
        self.mask = np.delete(self.mask, locats)
        # intersection of voxels and nonzero
        indices = np.where(np.in1d(self.voxels, locats))[0]
        # Cut whatever is not in the voxels.
        self.bixels = np.delete(self.bixels, indices)
        self.voxels = np.delete(self.voxels, indices)
        self.Dijs = np.delete(self.Dijs, indices)

    def removebixels(self, pitch):
        bixelkill = np.where(0 != (self.bixels % pitch) )
        bixelkill = np.where(self.bixels < 60)
        self.bixels = np.delete(self.bixels, bixelkill)
        self.smallvoxels = np.delete(self.smallvoxels, bixelkill)
        self.Dijs = np.delete(self.Dijs, bixelkill)
        self.mask = self.mask[np.unique(self.smallvoxels)]

    ## Read Weiguo's Case
    def readWeiguosCase(self):
        # Assign structures and thresholds for each of them in order of how important they are
        if "Prostate" == tumorsite:
            self.OARList = [21, 6, 11, 13, 14, 8, 12, 15, 7, 9, 5, 4, 20, 19, 18, 10, 22, 10, 11, 17, 12, 3, 15, 16, 9, 5, 4, 20, 21, 19]
            self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 78, 10, 10, 10, 10, 10, 10, 10, 10, 1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            self.TARGETList = [2]
            self.TARGETThresholds = [78]
        elif "Lung" == tumorsite:
            self.OARList = [5, 4, 7, 3, 2, 13, 6, 1]
            self.OARThresholds = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
            self.TARGETList = [11, 12]
            self.TARGETThresholds = [70, 70]
        else:
            self.OARList = [4]
            self.OARThresholds = [1]
            self.TARGETList = [8, 7]
            self.TARGETThresholds = [35, 45]
        dtype=np.uint32

        self.bixels = getvector(self.base_dir + 'dij/Bixels_out.bin', np.int32)
        self.voxels = getvector(self.base_dir + 'dij/Voxels_out.bin', np.int32)
        self.Dijs = getvector(self.base_dir + 'dij/Dijs_out.bin', np.float32)
        self.ALLList = self.TARGETList + self.OARList
        # get subsample mask (img_arr will have 1 in the positions where there is data)
        img_arr = getvector(self.base_dir + self.img_filename, dtype=dtype)
        # Only use a subsample of img_arr
        if do_subsample:
            img_arr = get_sub_sub_sample(img_arr, self.maxvoxels)
        # get structure file (used for the mask)
        struct_img_arr = getvector(self.base_dir + self.struct_img_filename, dtype=dtype)
        # Convert the mask into a list of unitary structures. A voxel gets assigned to only one place
        img_struct = get_structure_mask(reversed(self.ALLList), struct_img_arr)
        # Get the subsampled list of voxels
        self.mask = get_subsampled_mask(img_struct, img_arr)
        # Select only the voxels that exist in the small voxel space provided.
        if tumorsite == "Prostate":
            self.removezeroes([0, 18])
        else:
            self.removezeroes([0, 10, 14, 15, 8, 16, 9, 17])

    def maxTgtDoses(self, numProjections, k10):
        # This function will calculate the maximum bixel to a target coming from a particular beamlet
        bdoses = np.zeros(self.L, numProjections)
# Plot a few histograms again:
def histogramPlotter(abc, thismodel):
    leavelengths = abc['leavelengths']
    t = abc['t']
    plt.clf()
    binsequence = [i for i in np.arange(min(leavelengths), max(leavelengths), 0.01)] + [max(leavelengths)]
    plt.hist(np.array(leavelengths), bins=binsequence)
    # Add a few extra ticks to the labels
    plt.xlabel('Leaf Opening Times')
    #plt.text(abc['minLength'], 7, str(abc['minLength'])[0:6], color='r', rotation=89)
    #plt.text(abc['avLength'], 7, str(abc['avLength'])[0:6], color='r', rotation=89)
    plt.title('histogram ' + thismodel + ' model with Min. LOT:' + str(round(abc['minLength'], 2)) + ' and Average LOT:' + str(round(abc['avLength'], 2)))
    chunkName = 'minLOT' + thismodel + str(round(abc['minLength'], 2)).replace(".", "_") + 'avgLOT' + str(round(abc['avLength'], 2)).replace(".", "_")
    plt.savefig(outputDirectory + 'histogram' + chunkName + '.pdf', format = 'pdf')

def counterBlinker(ll):
    return(len(ll['leavelengths']))


p40 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-fullModel-MinLOT-0.04-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
p20 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))

print('40 avgLOT:', p40['avLength'], '20 avgLOT:', p20['avLength'])

p1 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700.pkl", "rb"))
p2 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
p3 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-fullModel-MinLOT-0.04-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
p4 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-pairModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700.pkl", "rb"))
p5 = pickle.load(open("outputMultiProj/pickleresults-Prostate-153-pairModel-MinLOT-0.02-minAvgLot-0.17-vxls-16686-ntnsty-700.pkl", "rb"))
p6 = pickle.load(open("outputMultiProj/pickleresults-Prostate-153-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-8344-ntnsty-700.pkl", "rb"))
p7 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-IMRT-MinLOT-0.02-minAvgLot-0.077-vxls-16677-ntnsty-700.pkl", "rb"))
p8 = pickle.load(open("outputMultiProj/pickleresults-Prostate-51-IMRT20msec-MinLOT-0.02-minAvgLot-0.077-vxls-16677-ntnsty-700.pkl", "rb"))

print(counterBlinker(p1))
print(counterBlinker(p4))
print(counterBlinker(p7))
print(counterBlinker(p8))
#histogramPlotter(p1, 'detailed')
#histogramPlotter(p2, 'detailed')
#histogramPlotter(p3, 'detailed')
#histogramPlotter(p4, 'odd-even')
#histogramPlotter(p5, 'odd-even')
histogramPlotter(p6, 'detailed')
# Plot the dose volume histogram
versus = ''
def plotDVH(plotList, ct1, noPlot, secondPlot, doublePlot, titlestring):
    if not doublePlot:
        plt.clf()
    mycolors = list(mcolors.TABLEAU_COLORS)
    data = ct1['data']
    z = ct1['z_output']
    voxDict = {}
    data.TARGETList = np.intersect1d(np.array(data.TARGETList), np.unique(data.mask))
    data.TARGETList = np.intersect1d(np.array(data.TARGETList), plotList)
    data.OARList = np.intersect1d(np.array(data.OARList), np.unique(data.mask))
    data.OARList = np.intersect1d(np.array(data.OARList), plotList)
    for t in data.TARGETList:
        voxDict[t] = np.where(data.mask == t)[0]
    for o in data.OARList:
        voxDict[o] = np.where(data.mask == o)[0]
    dose = np.array([z[j] for j in range(data.totalsmallvoxels)])
    i = 0
    for index, sValues in voxDict.items():
        sVoxels = sValues
        hist, bins = np.histogram(dose[sVoxels], bins=100)
        dvh = 1. - np.cumsum(hist) / float(sVoxels.shape[0])
        dvh = np.insert(dvh, 0, 1)
        if noPlot:
            if secondPlot:
                plt.plot(bins, dvh, '-.', c=mycolors[i + 3], linewidth=2)
            else:
                plt.plot(bins, dvh, ':', c = mycolors[i + 3], linewidth=3)
        else:
            plt.plot(bins, dvh, c = mycolors[i + 3], linewidth=1, label=data.AllDict[index])
        i+=1
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    if doublePlot:
        if noPlot:
            pass
        else:
            #plt.title('Comparison of DVH plots: Odd-Even model vs. Detailed model')
            plt.title(titlestring)
            #plt.annotate('Continuous: 40 millisecond. Dotted: 20 millisecond', xy=(2, -0.035))
            #plt.annotate('Continuous: Detailed, Dotted: Simple', xy=(2, -0.035))
    else:
        plt.title('DVH-' + tumorsite + 'min. LOT = ' + str(data.timeM) + ' and min.AvgLOT = ' + str(data.timeA))
    print('printing DVH')
    if noPlot:
        pass
    else:
        first_legend = plt.legend(bbox_to_anchor=(0.48, 0.59), prop={'size': 9})
        plt.gca().add_artist(first_legend)
        if doublePlot:
            plt.savefig(outputDirectory + 'DVH2.pdf', format = 'pdf')
        else:
            plt.savefig(outputDirectory + 'DVH1.pdf', format = 'pdf')
        plt.close()

def plotDVHcompare(plotList, ct1, ct2, titlestring = 'Comparison of different Minimum LOT constraints on treatment quality'):
    plt.clf()
    plotDVH(plotList, ct1, True, False, True, titlestring)
    plotDVH(plotList, ct2, False, False, True, titlestring)

def compareSinogram():
    pass
    return(None)

tumorsite = "Prostate"
plotList = [6, 7, 8, 10, 13, 14, 2]
ct = pickle.load(open("outputMultiProj/calculateTProstate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700.pkl", "rb"))

# Compare DVH side to side:
#ct2 = pickle.load(open("outputMultiProj/calculateTProstate-51-IMRT-MinLOT-0.02-minAvgLot-0.077-vxls-16677-ntnsty-700.pkl", "rb"))
ct2 = pickle.load(open("outputMultiProj/calculateTProstate-51-pairModel-MinLOT-0.02-minAvgLot-0.17-vxls-16677-ntnsty-700.pkl", "rb"))
titlestring = "'DVH-' + tumorsite + 'min. LOT = ' + str(data.timeM) + ' and min.AvgLOT = ' + str(data.timeA)"
ctm1 = pickle.load(open("outputMultiProj/calculateTProstate-51-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
ctm2 = pickle.load(open("outputMultiProj/calculateTProstate-51-pairModel-MinLOT-0.03-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
ctm3 = pickle.load(open("outputMultiProj/calculateTProstate-51-fullModel-MinLOT-0.04-minAvgLot-0.17-vxls-8340-ntnsty-700.pkl", "rb"))
ctmIMRT = pickle.load(open("outputMultiProj/calculateTProstate-51-IMRT-MinLOT-0.02-minAvgLot-0.077-vxls-16677-ntnsty-700.pkl", "rb"))
ctSimple153 = pickle.load(open("outputMultiProj/calculateTProstate-153-fullModel-MinLOT-0.02-minAvgLot-0.17-vxls-16686-ntnsty-700.pkl", "rb"))
ctDetailed153 = pickle.load(open("outputMultiProj/calculateTProstate-153-pairModel-MinLOT-0.02-minAvgLot-0.17-vxls-16686-ntnsty-700.pkl", "rb"))

titlestring = 'DVH FMO Treatment'
#plotDVH(plotList, ctmIMRT, False, False, False, titlestring)
#titlestring = 'DVH FMO Treatment vs DVH min LOT 170msec Treatment'
#plotDVHcompare(plotList, ctmIMRT, ct2, titlestring)
titlestring = 'DVH of Simple and Detailed Models 153 Projections per Rotation.'
print('start DVH compare')
plotDVHcompare(plotList, ctDetailed153, ctSimple153, titlestring)
print('end DVH compare')
sys.exit()
def plotDVHtriplecompare(plotList, ctm1, ctm2, ctm3):
    plt.clf()
    titlestring = 'Comparison of different Minimum LOT constraints on treatment quality'
    plotDVH(plotList, ctm1, True, False, True, titlestring)
    plotDVH(plotList, ctm2, True, True, True, titlestring)
    plotDVH(plotList, ctm3, False, False, True, titlestring)

#plotDVHtriplecompare(plotList, ctm1, ctm2, ctm3)
plotDVHcompare(plotList, ctm1, ctm3)
print('done')
counter = 0
allens = 0.0
for row in ctmIMRT['tim']:
    for elem in row:
        if elem > 0.00000001:
            counter+=1
            allens+=elem
print('average', allens / counter)