import numpy as np
import time
import random
from readMnist import readMnistImage, readMnistLable

def predictAccurary(testLabel, predictions):
	result = 0
	for i in range(len(testLabel)):
		if testLabel[i] == predictions[i]:
			result = result + 1
	return result

def cutLables(lables):
	for i in range(len(lables)):
		if lables[i] == 0:
			lables[i] = 1
		else:
			lables[i] = -1
	return lables

def sgdSVM(trainData, trainLabel, testData, testLabel, maxIter, tolerance):
	iterr = 0
	batch_size = 500
	cur_sindex = 0
	loss = 10000
	eta = 0.001
	lambC = 20
	# w = np.zeros((784,1))
	w = np.random.random((784,1))
	b = np.random.random(1)
	while iterr < maxIter or loss > tolerance:
		diff_w = np.zeros((784,1))
		diff_b = 0
		for i in range(batch_size):
			index = cur_sindex + i
			data = trainData[index, :]
			data.shape = (784,1)
			if trainLabel[index] * (np.dot(w.transpose(), data) + b) >= 1:
				continue
			else:
				diff_w += w - lambC * trainLabel[index] * data
				diff_b -= lambC * trainLabel[index]
		w -= eta * diff_w
		b -= eta * diff_b
		# compute loss
		loss = 0
		for i in range(batch_size):
			index =  cur_sindex + i
			data = trainData[index, :]
			data.shape = (784,1)
			loss += max(0, 1 - trainLabel[index] * (np.dot(w.transpose(), data) + b))
		if iterr % 500 == 0:
			print(iterr, loss)
		cur_sindex = (cur_sindex + batch_size) % 60000
		# print(cur_sindex)
		# if cur_sindex == 0:
		# 	batch_size = min(batch_size * 10, 1000)
		# 	eta = max(eta / 10.0, 0.0001)
		iterr += 1
	print(iterr, loss)

		

print(time.localtime())
trainImage = readMnistImage('train-images.idx3-ubyte')
trainLabel = readMnistLable('train-labels.idx1-ubyte')
trainLabel = cutLables(trainLabel)
testImage = readMnistImage('t10k-images.idx3-ubyte')
testLabel = readMnistLable('t10k-labels.idx1-ubyte')
testLabel = cutLables(testLabel)
sgdSVM(trainData=trainImage, trainLabel=trainLabel, testData=testImage, testLabel=testLabel, maxIter=10000, tolerance=1)