import numpy as np
import heapq
import time
from collections import Counter

def distance(a, b):
	return np.linalg.norm(a-b)

def readMnistImage(filename):
	file_object = open(filename, 'rb')
	allText = file_object.read()
	file_object.close()
	mat = np.asarray([allText[i] for i in range(16,len(allText))])
	mat.shape = (int(len(allText)/784), 784)
	return mat

def readMnistLable(filename):
	file_object = open(filename, 'rb')
	allText = file_object.read()
	file_object.close()
	mat = np.asarray([allText[i] for i in range(8, len(allText))])
	return mat

def predictAccurary(testLabel, predictions):
	result = 0
	for i in range(len(testLabel)):
		if testLabel[i] == predictions[i]:
			result = result + 1
	return result

def knnPredict(K, trainImage, trainLabel, testImage):
	result = []
	for i in range(len(testImage)):
		if i % 100 == 0:
			print(i, time.localtime())
		# distances = [distance(trainImage[j], testImage[i]) for j in range(len(trainImage))]
		# distances = sorted(distances)
		# print(distances[:K])
		# print(i, time.localtime())
		# print(len(trainImage))
		indexes = [p[1] for p in heapq.nsmallest(K, [(distance(trainImage[j], testImage[i]), j) for j in range(len(trainImage))])]
		# print(i, time.localtime())
		labels = [trainLabel[i] for i in indexes]
		# print(labels)
		predict = [p[1] for p in heapq.nlargest(1, [(labels.count(j), j) for j in range(10)])]
		# print(predict)
		result = np.append(result, predict)
	return result

K = 1
print(time.localtime())
trainImage = readMnistImage('train-images.idx3-ubyte')
trainLabel = readMnistLable('train-labels.idx1-ubyte')
testImage = readMnistImage('t10k-images.idx3-ubyte')
testLabel = readMnistLable('t10k-labels.idx1-ubyte')
predictions = knnPredict(K, trainImage, trainLabel, testImage)
print('Here is the result: K = ' + str(K) + '  with Euclidean distance. Accuracy: ' + str(predictAccurary(testLabel, predictions)))
