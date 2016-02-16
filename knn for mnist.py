import numpy as np
import heapq
from collections import Counter

def convertByteToInt(bytes):
	result = 0
	for i in range(len(bytes)):
		result = (result << 8) + bytes[i]
	return result

def distanceEuclidean(a, b):
	c = [(a[i] - b[i]) for i in a]
	return np.linalg.norm(c)

def readMnistImage(filename):
	file_object = open(filename, 'rb')
	allText = file_object.read()
	m = convertByteToInt(allText[8:12])
	n = convertByteToInt(allText[12:16])
	result = [allText[16 + i*m*n: 16 + (i+1)*m*n] for i in range(int(len(allText)/(m*n)))]
	return result

def readMnistLable(filename):
	file_object = open(filename, 'rb')
	allText = file_object.read()
	return allText[8:]

def predictAccurary(testLabel, predictions):
	result = 0
	for i in range(len(testLabel)):
		if testLabel[i] == predictions[i]:
			result = result + 1
	return result

def knnPredict(K, trainImage, trainLabel, testImage):
	result = []
	for i in range(len(testImage)):
		print(i)
		indexes = [p[1] for p in heapq.nsmallest(K, [(distanceEuclidean(trainImage[j], testImage[i]), j) for j in range(len(trainImage))])]
		labels = [trainLabel[i] for i in indexes]
		predict = [p[1] for p in heapq.nlargest(1, [(labels.count(j), j) for j in range(10)])]
		result = np.append(result, predict)
	return result

K = 100
trainImage = readMnistImage('train-images.idx3-ubyte')
trainLabel = readMnistLable('train-labels.idx1-ubyte')
testImage = readMnistImage('t10k-images.idx3-ubyte')
testLabel = readMnistLable('t10k-labels.idx1-ubyte')
predictions = knnPredict(K, trainImage, trainLabel, testImage)
print('Here is the result: K = ' + str(K) + '  with Euclidean distance. Accuracy: ' + str(predictAccurary(testLabel, predictions)))
