import numpy as np
import os

def read_data(in_csv, dtype=np.float32):
	data = []
	with open(in_csv, "r") as inf:
		feats = inf.readline().strip().replace(" ", "").split(',')
		for line in inf:
			data.append(line.strip().split(","))

	data = np.asarray(data, dtype=dtype)
	return feats, data


def read_label(in_csv, n_class=2):
	labels = np.zeros((0, n_class))
	with open(in_csv, "r") as inf:
		inf.readline()
		for line in inf:
			label = int(line.strip())
			tmp_label = np.zeros((1, n_class))
			tmp_label[0][label] = 1
			labels = np.vstack((labels, tmp_label))

	return labels


def split_train_val(data, labels, val_indices):
	num_data = data.shape[0]
	train_indices = np.asarray([idx for idx in range(num_data) if idx not in val_indices])

	if len(val_indices) == 0:	
		return data[train_indices], None, labels[train_indices], None
	else:
		return data[train_indices], data[val_indices], labels[train_indices], labels[val_indices]


def gen_val_indices(in_csv, k_fold):
	with open(in_csv, "r") as inf:
		lines = inf.readlines()
		num_data = len(lines)-1	# Minus the header

	name, ext = os.path.splitext(in_csv)

	val_indices = []
	fold_size = int(num_data / k_fold)

	indices = np.random.permutation(num_data)
	indice_path = name + "_" + str(k_fold) + "fold"
	with open(indice_path, "w") as outf:
		for i in range(k_fold):
			if i == k_fold - 1:
				start, end = i * fold_size, num_data
			else:
				start, end = i * fold_size, (i+1) * fold_size

			val_indices.append(np.asarray(indices[start:end], dtype=np.int32))
			for indice in indices[start:end]:
				outf.write(str(indice) + " ")
			outf.write("\n")

	return val_indices 


def get_val_indices(indice_path):
	val_indices = []
	with open(indice_path, "r") as inf:
		for line in inf:
			val_indices.append(np.asarray(line.strip().split(" "), dtype=np.int32))

	return val_indices
