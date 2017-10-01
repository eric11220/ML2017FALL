import os
import sys
import math
import numpy as np
from splitData import *

COEF_DIR = '../coefficients'

def rmse(logits, labels):
	return math.sqrt(np.mean(np.power(logits - labels, 2)))

def predict(coef, train_data):
	return np.sum(coef * train_data, axis=1)

def linear_reg(train_data, train_label, val_data, val_label, n_epoch=10000, lr=5e-4, batch_size=1, display_epoch=10):
	num_data = train_data.shape[0]
	train_data = np.concatenate((train_data, np.ones((num_data, 1))), axis=1)
	
	if val_data is not None:
		val_data = np.concatenate((val_data, np.ones((val_data.shape[0], 1))), axis=1)

	dim = train_data.shape[1]
	coef = np.zeros(dim)

	for epoch in range(n_epoch):
		batch_idx = 0

		indices = np.random.permutation(num_data) 
		train_data = train_data[indices]
		train_label = train_label[indices]

		while batch_idx < num_data:
			batch_data = train_data[batch_idx:batch_idx+batch_size, :]
			batch_label = train_label[batch_idx:batch_idx+batch_size]
			logits = predict(coef, batch_data)

			error = rmse(logits, batch_label)
			coef += -(1/error) * np.dot( (logits - batch_label), batch_data) / batch_data.shape[0] * 2 * lr

			batch_idx += batch_size

		if epoch % 10 == 0 and val_data is not None:
			val_logits = predict(coef, val_data)
			print('>epoch=%d, lrate=%.5f, error=%.3f, validation error=%.6f' % (epoch, lr, error, rmse(val_logits, val_label)))
		else:
			print('>epoch=%d, lrate=%.5f, error=%.3f' % (epoch, lr, error))

	return coef

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python linearReg.py input_csv k_fold")
		exit()

	in_csv = sys.argv[1]
	k_fold = int(sys.argv[2])

	data_dir = os.path.dirname(in_csv)
	first_fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_0")
	if not os.path.isfile(first_fold_path):
		gen_val_indices(in_csv, k_fold)
	
	if k_fold > 1:
		for i in range(k_fold):
			fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_" + str(i))
			train_data, val_data, train_lbl, val_lbl = split_data(in_csv, fold_path)

			train_mean = np.mean(train_data, axis=0)
			train_std = np.std(train_data, axis=0)
			train_data = (train_data - train_mean) / train_std
			val_data = (val_data - train_mean) / train_std

			coef = linear_reg(train_data, train_lbl,val_data, val_lbl,  batch_size=1, n_epoch=100)
	else:
		fold_path = os.path.join(data_dir, "indices1_0")
		train_data, val_data, train_lbl, val_lbl = split_data(in_csv, fold_path)

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		train_data = (train_data - train_mean) / train_std

		coef = linear_reg(train_data, train_lbl, val_data, val_lbl, batch_size=1, n_epoch=100)

	for i in range(k_fold):
		coef_path = os.path.join()  
		
if __name__ == '__main__':
	main()
