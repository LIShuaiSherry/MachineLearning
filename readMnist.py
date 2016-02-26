import numpy as np

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