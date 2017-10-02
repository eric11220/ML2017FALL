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

def ada_grad(prev_grad, grad):
	sqr_grad = grad ** 2
	prev_grad += sqr_grad

	mod_grad = grad / np.sqrt(prev_grad)
	return prev_grad, mod_grad

def linear_reg(train_data, train_label, val_data, val_label, n_epoch=10000, lr=1, batch_size=1, display_epoch=10, lamb=0.1, ada=True):
	num_data = train_data.shape[0]
	train_data = np.concatenate((train_data, np.ones((num_data, 1))), axis=1)
	
	if val_data is not None:
		val_data = np.concatenate((val_data, np.ones((val_data.shape[0], 1))), axis=1)

	dim = train_data.shape[1]
	coef = np.zeros(dim)
	prev_grad = np.zeros(dim)

	for epoch in range(n_epoch):
		batch_idx = 0

		indices = np.random.permutation(num_data) 
		train_data = train_data[indices]
		train_label = train_label[indices]

		while batch_idx < num_data:
			batch_data = train_data[batch_idx:batch_idx+batch_size, :]
			batch_label = train_label[batch_idx:batch_idx+batch_size]
			logits = predict(coef, batch_data)

			loss = rmse(logits, batch_label)
			regularization = 0.5 * lamb * np.sum(coef ** 2)
			'''
			print( ((1/loss) * np.dot( (logits - batch_label), batch_data)  / batch_data.shape[0] * 2) * (lamb * coef) > 0 )
			input("")
			'''

			grad = ((1/loss) * np.dot( (logits - batch_label), batch_data) + lamb * coef)  / batch_data.shape[0]
			if ada is True:
				prev_grad, grad = ada_grad(prev_grad, grad)
			coef -= grad * lr

			batch_idx += batch_size

		if epoch % display_epoch == 0:
			train_logits = predict(coef, train_data)
			if val_data is not None:
				val_logits = predict(coef, val_data)
				print('>epoch=%d, lrate=%.5f, error=%.3f, validation error=%.6f' % (epoch, lr, rmse(train_logits, train_label), rmse(val_logits, val_label)))
			else:
				print('>epoch=%d, lrate=%.5f, error=%.3f' % (epoch, lr, rmse(train_logits, train_label)))

	if val_data is not None:
		val_logits = predict(coef, val_data)
		return coef, rmse(val_logits, val_label)
	else:
		return coef, None

def main():
	argc = len(sys.argv)
	if argc != 7:
		print("Usage: python linearReg.py input_csv k_fold n_epoch lambda ada lr")
		exit()

	in_csv = sys.argv[1]
	k_fold = int(sys.argv[2])
	n_epoch = int(sys.argv[3])
	lamb = float(sys.argv[4])
	lr = float(sys.argv[6])

	if sys.argv[5] == '1':
		ada = True
	else:
		ada = False

	data_dir = os.path.dirname(in_csv)
	first_fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_0")
	if not os.path.isfile(first_fold_path):
		gen_val_indices(in_csv, k_fold)
	
	sum_error = 0
	for i in range(k_fold):
		fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_" + str(i))
		train_data, val_data, train_lbl, val_lbl, feat_order = split_data(in_csv, fold_path)

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		train_data = (train_data - train_mean) / train_std

		if val_data is not None:
			val_data = (val_data - train_mean) / train_std

		coef, error = linear_reg(train_data, train_lbl,val_data, val_lbl,  batch_size=1, n_epoch=n_epoch, lamb=lamb, ada=ada, lr=lr)
		if error is not None:
			sum_error += error

		csv_dir = os.path.dirname(in_csv).split("/")[-1]
		coef_path = os.path.join(COEF_DIR, csv_dir + "_" + str(k_fold) + '_' + str(i))
		with open(coef_path, 'w') as outf:
			for feat in feat_order:
				outf.write(feat + " ")
			outf.write("\n")

			for val in train_mean:
				outf.write(str(val) + " ")
			outf.write("\n")
			for val in train_std:
				outf.write(str(val) + " ")
			outf.write("\n")

			for c in coef:
				outf.write(str(c) + " ")
			outf.write("\n")

	if  k_fold > 1:
		print(sum_error / k_fold)

if __name__ == '__main__':
	main()
