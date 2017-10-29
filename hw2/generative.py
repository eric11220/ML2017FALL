import sys
import os
import sys
import math
import numpy as np
from splitData import *

def predict(X_test, mu1, mu2, shared_cov, N1, N2):
	sigma_inv = np.linalg.inv(shared_cov)
	w = np.dot( (mu1 - mu2), sigma_inv)
	x = X_test.T
	b = -0.5 * np.dot(np.dot([mu1], sigma_inv), mu1) \
		+ 0.5 * np.dot(np.dot([mu2], sigma_inv), mu2) \
		+ np.log(float(N1)/N2)

	a = np.dot(w, x) + b
	y = sigmoid(a)
	return y


def sigmoid(x):
	res = 1 / (1 + np.exp(-x))
	return np.clip(res, 0.0000000000001, 0.999999999999)

argc = len(sys.argv)
if argc != 5:
	print("Usage: python train.py X_train Y_train X_test out_csv")
	exit()

X_train = sys.argv[1]
Y_train = sys.argv[2]
X_test = sys.argv[3]
csv_path = sys.argv[4]

# Read training data
_, X_train = read_data(X_train)
_, Y_train = read_data(Y_train, dtype=np.int16)

# Read testing data
_, X_test = read_data(X_test)

		
train_data_size, dim = X_train.shape
cnt1, cnt2 = 0, 0
		
mu1 = np.zeros((dim,))
mu2 = np.zeros((dim,))
for i in range(train_data_size):
	if Y_train[i] == 1:
		mu1 += X_train[i]
		cnt1 +=1
	else:
		mu2 += X_train[i]
		cnt2 +=1

mu1 /= cnt1
mu2 /= cnt2


sigma1 = np.zeros((dim, dim))
sigma2 = np.zeros((dim, dim))
for i in range(train_data_size):
	if Y_train[i] == 1:
		sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
	else:
		sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])

sigma1 /= cnt1
sigma2 /= cnt2


shared_sigma = (float(cnt1) / train_data_size) * sigma1 \
			   + (float(cnt2) / train_data_size) * sigma2

Xtest_pred = predict(X_test, mu1, mu2, shared_sigma, cnt1, cnt2)
Xtest_pred = np.around(Xtest_pred)

with open(csv_path, "w") as outf:
	outf.write("id,label\n")
	for idx, label in enumerate(Xtest_pred):
		outf.write(str(idx+1) + ',' + str(int(label)) + "\n")
