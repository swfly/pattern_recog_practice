import numpy as np
import struct


#samples inputs are 28x28 byte numpy arrays
#labels are bytes indicating digit (0~9)
samples=[]
sampleLabels=[]
testSamples=[]
testLabels=[]

#read data as bytes
with open('data.ubyte', mode='rb') as file:
    imageData = file.read()
with open('labels.ubyte', mode='rb') as file:
    labelData = file.read()


#parse data
dataStartOffset = 16
labelStartOffset = 8
dataNumber = struct.unpack(">i",imageData[4:8])[0]
labelNumber = struct.unpack(">i",labelData[4:8])[0]


for i in range(dataNumber):
    dat = np.zeros((28,28),dtype=np.uint8)
    idx = 0
    for x in range(28):
        for y in range(28):
            dat[x,y] = imageData[dataStartOffset+i*28*28+idx]
            idx = idx+1
    if i < labelNumber:
        res = labelData[labelStartOffset+i]
        samples.append(dat)
        sampleLabels.append(res)
    else:
        testSamples.append(dat)
#parse done

#implement your k-NN here
def knn(inputSample):
    return -1

#process all the test inputs
for s in testSamples:
    res = knn(s)
    testLabels.append(res)