import numpy as np
import os


def padd_zero(data, max_len):
	dup = []
	for row in data:
		length = len(row)

		tmp_row = np.array(row)
		for _ in range(length, max_len):
			tmp_row = np.append(tmp_row, 0)

		dup.append(tmp_row)

	return np.array(dup, dtype=int)


def read_data(data_csv, label_csv, word_index, handle_oov=False, one_hot=True):
	vocab_size = len(word_index)

	if label_csv is not None:
		label_inf = open(label_csv, 'r')

	data, labels, lens = [], [], []
	with open(data_csv, 'r') as inf:
		for line in inf:
			if label_csv is not None:
				label = int(label_inf.readline().strip())

			row_data = []
			line = line.strip().split(' ')
			for element in line:
				index = word_index.get(element, None)
				if index is None:
					if handle_oov is True:
						row_data.append(vocab_size + 1)
					else:
						continue
				else:
					row_data.append(index)

			length = len(row_data)
			if length > 0:
				lens.append(length)

				data.append(row_data)
				if label_csv is not None:
					labels.append(label)

	if one_hot is True and label_csv is not None:
		from keras.utils import to_categorical
		labels = to_categorical(np.asarray(labels, dtype=int))
	return np.array(data, dtype=object), labels, np.asarray(lens)


def split_train_val(data, labels, val_indices):
	num_data = data.shape[0]
	train_indices = np.asarray([idx for idx in range(num_data) if idx not in val_indices])

	if len(val_indices) == 0:	
		return data[train_indices], None, labels[train_indices], None
	else:
		return data[train_indices], data[val_indices], labels[train_indices], labels[val_indices]


def gen_val_indices(in_csv, k_fold, indice_path):
	with open(in_csv, "r") as inf:
		lines = inf.readlines()
		num_data = len(lines)

	name, ext = os.path.splitext(in_csv)

	val_indices = []
	fold_size = int(num_data / k_fold)

	indices = np.random.permutation(num_data)
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
