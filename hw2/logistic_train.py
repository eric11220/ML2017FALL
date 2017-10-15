import os
import sys
import math
import numpy as np
from splitData import *

np.set_printoptions(suppress=True)

COEF_DIR = 'coefficients'

num_losses = 10
tolerance = 0.001
epsilon = 1e-8


def cross_entropy(logits, labels):
	return np.mean(-(labels* np.log(logits+epsilon) + (1-labels) * np.log((1-logits+epsilon))))


def predict(coef, train_data):
	return 1 / (1 + np.exp(-np.sum(coef * train_data, axis=1)))


def ada_grad(prev_grad, grad):
	sqr_grad = grad ** 2
	prev_grad += sqr_grad

	mod_grad = grad / (np.sqrt(prev_grad) + 1e-8)
	return prev_grad, mod_grad


def adam(prev_grad, prev_divisor, grad, iter, beta1=0.9, beta2=0.99):
	prev_grad = beta1 * prev_grad + (1-beta1) * grad
	prev_divisor = beta2 * prev_divisor + (1-beta2) * (grad ** 2)

	dividend = prev_grad / (1 - beta1**iter)
	divisor = prev_divisor / (1 - beta2**iter)

	mod_grad = dividend / (np.sqrt(divisor) + 1e-8)

	return prev_grad, prev_divisor, mod_grad 


def get_acc(logits, labels):
	logits = logits > 0.5
	return np.sum(logits == labels) / len(labels)


def linear_reg(train_data, train_label, val_data, val_label, n_epoch=10000, lr=1, batch_size=1, display_epoch=10, lamb=0.1, gd_alg="gd", early_stop=False):
	num_data = train_data.shape[0]

	train_data = np.concatenate((train_data, np.ones((num_data, 1))), axis=1)
	if val_data is not None:
		val_data = np.concatenate((val_data, np.ones((val_data.shape[0], 1))), axis=1)

	train_label = np.reshape(train_label, [-1])
	val_label = np.reshape(val_label, [-1])

	dim = train_data.shape[1]
	coef = np.zeros(dim)
	old_coef = np.copy(coef)
	prev_grad, prev_divisor = np.zeros(dim), np.zeros(dim)

	if early_stop is True:
		n_epoch = 20000

	epoch, loss_list = 0, np.asarray([])
	for epoch in range(n_epoch):
		batch_idx, iter = 0, 1

		indices = np.random.permutation(num_data) 
		train_data = train_data[indices]
		train_label = train_label[indices]

		while batch_idx < num_data:
			batch_data = train_data[batch_idx:batch_idx+batch_size, :]
			batch_label = train_label[batch_idx:batch_idx+batch_size]
			logits = predict(coef, batch_data)

			loss = cross_entropy(logits, batch_label)
			regularization = 0.5 * lamb * np.sum(coef ** 2)

			grad = (np.dot( (logits - batch_label), batch_data))  / batch_data.shape[0] + lamb * coef
			if gd_alg == 'ada':
				prev_grad, grad = ada_grad(prev_grad, grad)
			elif gd_alg == 'adam':
				prev_grad, prev_divisor, grad = adam(prev_grad, prev_divisor, grad, iter)
			coef -= grad * lr

			batch_idx += batch_size
			iter += 1

		if epoch % display_epoch == 0:
			train_logits = predict(coef, train_data)
			train_loss = cross_entropy(train_logits, train_label)
			train_acc = get_acc(train_logits, train_label)

			if early_stop is True:
				loss_list = np.append(loss_list, train_loss)
				if len(loss_list) <= 1:
					mean_loss_diff = 0
				elif len(loss_list) < num_losses:
					mean_loss_diff = np.mean(np.abs(loss_list[1:] - loss_list[:-1]))
				else: 
					loss_list = loss_list[1:]
					mean_loss_diff = np.mean(np.abs(loss_list[1:] - loss_list[:-1]))
					if mean_loss_diff < tolerance:
						print("Converged on criteria: differences on %d losses smaller than %f" %(num_losses, tolerance))
						break
			else:
				mean_loss_diff = 0

			if val_data is not None:
				val_logits = predict(coef, val_data)
				val_loss = cross_entropy(val_logits, val_label)
				val_acc = get_acc(val_logits, val_label)
				print('>epoch=%d, lrate=%.4f, acc=%.4f, error=%.4f, validation acc=%.4f, error=%.4f, mean %d-loss diff=%.4f' \
								% (epoch, lr, train_acc, train_loss, val_acc, val_loss, num_losses, mean_loss_diff))
			else:
				print('>epoch=%d, lrate=%.4f, acc=%.4f, error=%.4f, mean %d-loss diff=%.4f' \
								% (epoch, lr, train_acc, train_loss, num_losses, mean_loss_diff))

	if val_data is not None:
		val_logits = predict(coef, val_data)
		return coef, cross_entropy(val_logits, val_label)
	else:
		return coef, None


def parse_deg_from_filename(in_csv):
	if 'degree' not in in_csv:
		deg = 1
	else:
		name, ext = os.path.splitext(in_csv)
		deg = name.split('degree')[1]
	return deg


def write_coef_to_file(in_csv, k_fold, fold, deg, feat_order, train_mean, train_std, coef):
	in_csv = os.path.basename(in_csv)
	name, ext = os.path.splitext(in_csv)
	coef_path = os.path.join(COEF_DIR, name + "_" + str(k_fold) + '-' + str(fold) + "_deg" + str(deg))
	np.save(coef_path, [feat_order, train_mean, train_std, coef])


def main():
	argc = len(sys.argv)
	if argc != 9:
		print("Usage: python train.py X_train Y_train k_fold n_epoch lambda gd_alg lr batch_size")
		exit()

	X_train = sys.argv[1]
	Y_train = sys.argv[2]
	k_fold = int(sys.argv[3])
	n_epoch = int(sys.argv[4])
	lamb = float(sys.argv[5])
	lr = float(sys.argv[7])
	batch_size = int(sys.argv[8])
	gd_alg = sys.argv[6]

	deg = parse_deg_from_filename(X_train)

	if k_fold > 1:
		name, ext = os.path.splitext(X_train)
		indice_path = name + "_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(X_train, k_fold)
		else:
			indices = get_val_indices(indice_path)
	else:
		indices = [[]]

	feats, data = read_data(X_train)
	_, labels = read_data(Y_train, dtype=np.int16)

	sum_error = 0
	for i in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[i])

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		train_data = (train_data - train_mean) / train_std

		if val_data is not None:
			val_data = (val_data - train_mean) / train_std

		coef, error = linear_reg(train_data, train_lbl, val_data, val_lbl, \
					   	batch_size=batch_size, n_epoch=n_epoch, lamb=lamb, gd_alg=gd_alg, lr=lr, early_stop=False)

		if error is not None:
			sum_error += error
		else:
			print(error)

		write_coef_to_file(X_train, k_fold, i, deg, feats, train_mean, train_std, coef)

	if  k_fold > 1:
		sys.stderr.write(str(sum_error / k_fold) + "\n")


if __name__ == '__main__':
	main()
