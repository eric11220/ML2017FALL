from __future__ import print_function

import cv2
import scipy
import imutils
import numpy as np

from splitData import *

def flatten_matrix(matrix):
	vector = matrix.flatten(1)
	vector = vector.reshape(1, len(vector))
	return vector

def zca_whitening(inputs):
	sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
	U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
	epsilon = 0.1				#Whitening constant, it prevents division by zero
	ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)					 #ZCA Whitening matrix
	return np.dot(ZCAMatrix, inputs)   #Data whitening
	
def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
							  sqrt_bias=10, min_divisor=1e-8):

	assert X.ndim == 2, "X.ndim must be 2"
	scale = float(scale)
	assert scale >= min_divisor

	mean = X.mean(axis=1)
	if subtract_mean:
		X = X - mean[:, np.newaxis]  
	else:
		X = X.copy()
	if use_std:
		ddof = 1
		if X.shape[1] == 1:
			ddof = 0
		normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
	else:
		normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
	normalizers[normalizers < min_divisor] = 1.
	X /= normalizers[:, np.newaxis]  # Does not make a copy.
	return X
def ZeroCenter(data):
	data = data - np.mean(data,axis=0)
	return data

def normalize(arr):
	minval = min(arr)
	maxval = max(arr)

	arr = [(val - minval) / (maxval - minval) for val in arr]
	return arr

def Flip(data):
	dataFlipped = data[..., ::-1].reshape(2304).tolist()
	return dataFlipped

def Roated15Left(data):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(2304).tolist()

def Roated15Right(data):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -30, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(2304).tolist()

def Zoomed(data):
	datazoomed = scipy.misc.imresize(data,(60,60))
	datazoomed = datazoomed[5:53,5:53]
	datazoomed = datazoomed.reshape(2304).tolist()
	return datazoomed

def shiftedUp20(data):
	translated = imutils.translate(data, 0, -5)
	translated2 = translated.reshape(2304).tolist()
	return translated2
def shiftedDown20(data):
	translated = imutils.translate(data, 0, 5)
	translated2 = translated.reshape(2304).tolist()
	return translated2

def shiftedLeft20(data):
	translated = imutils.translate(data, -5, 0)
	translated2 = translated.reshape(2304).tolist()
	return translated2
def shiftedRight20(data):
	translated = imutils.translate(data, 5, 0)
	translated2 = translated.reshape(2304).tolist()
	return translated2

def outputImage(pixels,number):
	data = pixels
	name = str(number)+"output.jpg" 
	scipy.misc.imsave(name, data)

def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(list):
	Intonparray = np.asarray(list)
	data = Intonparray.reshape(48,48)
	data2 = ZeroCenter(data)
	data3 = zca_whitening(flatten_matrix(data2)).reshape(48,48)
	data4 = global_contrast_normalize(data3)
	data5 = np.rot90(data4,3)
	return data5

def convert_data(data, label, whitening=False):
	x, y = [], []

	for dat in data:
		dat = dat.tolist()
		if whitening is True:
			dat = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(dat)
		x.append(dat)

		y = label.tolist()

	# train_x: list of lists of 2304 pixel values
	# train_y: list of labels
	return x, y


