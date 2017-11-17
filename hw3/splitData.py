import numpy as np
import os

def nn_feature(model, data):

	from keras.models import Model
	from keras import backend as K

	model_part = K.function([model.layers[0].input, K.learning_phase()], 
							[model.layers[-2].output])

	num_data = len(data)
	idx, batch, layer_output = 0, 500, None
	while(idx < num_data):
		if idx + batch > num_data:
			end = num_data
		else:
			end = idx + batch

		result = model_part([data[idx:end], 0])[0]
		if layer_output is None:
			layer_output = result
		else:
			layer_output = np.concatenate((layer_output, result), axis=0)

		idx += batch

	return layer_output

def find_info(model_path):
	name, ext = os.path.splitext(model_path)
	fold_info = model_path.split('/')[-2]

	try:
		k_fold, fold = fold_info.split('-')
		k_fold, fold = int(k_fold), int(fold)
	except:
		k_fold, fold = None, None

	do_zca = "zca" in model_path
	return k_fold, fold, do_zca


def convert_to_svm_label(labels, n_class=7):
	tmp_y = []
	for label in labels:
		arr = np.asarray([-1 for _ in range(n_class)])
		arr[int(label)] = 1
		tmp_y.append(arr)
	return np.asarray(tmp_y)


def read_data(in_csv, num_class=7, one_hot_encoding=True, svm_label=False):
	data, labels = [], []
	with open(in_csv, "r") as inf:
		headers = inf.readline().strip().replace(" ", "").split(',')
		for line in inf:
			label, feats = line.strip().split(",")

			if one_hot_encoding is True:
				label = int(label)
				if svm_label is True:
					one_hot = np.asarray([-1 for _ in range(num_class)])
				else:
					one_hot = np.zeros((num_class,))

				one_hot[label] = 1
				labels.append(one_hot)
			else:
				labels.append(label)

			data.append(feats.split(" "))

	labels = np.asarray(labels, dtype=np.int16)
	data = np.asarray(data, dtype=np.float32)
	data /= 255.0
	return labels, data


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
