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

def Flip(data, dim):
	dataFlipped = data[..., ::-1].reshape(dim*dim).tolist()
	return dataFlipped

def Roated15Left(data, dim):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(dim*dim).tolist()

def Roated15Right(data, dim):
	num_rows, num_cols = data.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -30, 1)
	img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
	return img_rotation.reshape(dim*dim).tolist()

def Zoomed(data, dim):
	zoomed_dim = 1.25 * dim
	start_dim = int(0.125 * dim)
	end_dim = start_dim + dim

	datazoomed = scipy.misc.imresize(data,(zoomed_dim, zoomed_dim))
	datazoomed = datazoomed[start_dim:end_dim, start_dim:end_dim]
	datazoomed = datazoomed.reshape(dim*dim).tolist()
	return datazoomed

def shiftedUp20(data, dim):
	translated = imutils.translate(data, 0, -5)
	translated2 = translated.reshape(dim*dim).tolist()
	return translated2
def shiftedDown20(data, dim):
	translated = imutils.translate(data, 0, 5)
	translated2 = translated.reshape(dim*dim).tolist()
	return translated2

def shiftedLeft20(data, dim):
	translated = imutils.translate(data, -5, 0)
	translated2 = translated.reshape(dim*dim).tolist()
	return translated2
def shiftedRight20(data, dim):
	translated = imutils.translate(data, 5, 0)
	translated2 = translated.reshape(dim*dim).tolist()
	return translated2

def global_contrast_normalize(X, scale=1., sqrt_bias=10, min_divisor=1e-8):

	assert X.ndim == 2, "X.ndim must be 2"
	scale = float(scale)
	assert scale >= min_divisor

	mean = X.mean(axis=1)
	X = X - mean[:,np.newaxis]

	normalizers = np.sqrt(sqrt_bias + X.var(axis=1)) / scale
	normalizers[normalizers < min_divisor] = 1.

	X /= normalizers[:, np.newaxis]  # Does not make a copy.
	return X

def zca_whitening(data, zca_mat=None):
	#Correlation matrix
	sigma = np.dot(data.T, data)/(data.shape[0] - 1)

	#Singular Value Decompositionoo
	U, S, _ = np.linalg.svd(sigma)

	#Whitening constant, it prevents division by zero
	epsilon = 1e-6

	s = np.sqrt(S.clip(epsilon))
	s_inv = np.diag(1./s)
	s = np.diag(s)

	if zca_mat is None:
		zca_mat = np.dot(np.dot(U, s_inv), U.T)

	return np.dot(data, zca_mat.T), zca_mat

def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(data, zca_mat=None, train_mean=None):
	orig_shape = data.shape
	data = np.reshape(data, [data.shape[0], -1])

	if train_mean is None:
		train_mean = np.mean(data, axis=0)
	zero_centered = data - train_mean

	gcn = global_contrast_normalize(zero_centered)
	white_gcn, zca_mat = zca_whitening(gcn, zca_mat)
	white_gcn = np.reshape(white_gcn, orig_shape)

	return white_gcn, zca_mat, train_mean
