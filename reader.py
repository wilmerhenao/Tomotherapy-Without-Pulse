import pickle

outputDirectory = "outputMultiProj/"
yBar = 1000

output3 = open(outputDirectory + 'pickleresults-IMRTWithIntensity' + str(yBar) + '.pkl', 'wb')
myresults = pickle.load(output3)
output3.close()

print(myresults)