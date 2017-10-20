import numpy as np
import os

def read_indices(path):
	indices = {}
	with open(path, 'r') as inf:
		for line in inf:
			vals = line.strip().split(' ')
			month, val_indices = vals[0], np.asarray([int(idx) for idx in vals[1:]])
			indices[month] = val_indices

	return indices

def split_train_val(data, labels, num_data, val_indices):
	train_indices = np.asarray([idx for idx in range(num_data) if idx not in val_indices])

	if len(val_indices) == 0:	
		return data[train_indices], None, labels[train_indices], None
	else:
		return data[train_indices], data[val_indices], labels[train_indices], labels[val_indices]

def split_data(path, indices_path):
	val_indices = read_indices(indices_path)

	cur_month, data, labels = 0, [], []
	all_train_data, all_val_data = None, None
	all_train_lbl, all_val_lbl = None, None
	with open(path, 'r') as inf:
		lines = inf.readlines()

		# get feat order
		feat_order = lines[0].strip().split(' ')

		lines = lines[1:]
		for idx, line in enumerate(lines):
			vals = line.strip().split(" ")

			month = vals[0]
			instance = np.asarray(vals[1:-1], dtype=np.float32)
			label = vals[-1]

			if cur_month != month or idx == len(lines)-1:
				if idx == len(lines)-1:
					data.append(instance)
					labels.append(label)

				num_data = len(data)
				if num_data > 0:
					shape = [num_data, data[0].shape[0]]
					data = np.concatenate(data).reshape(shape)
					labels = np.asarray(labels, dtype=np.float32)

					train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, len(data), val_indices[cur_month])

					if all_train_data is None:
						all_train_data = train_data
						all_train_lbl = train_lbl

						all_val_data = val_data
						all_val_lbl = val_lbl
					else:
						all_train_data = np.vstack((all_train_data, train_data))
						all_train_lbl = np.concatenate((all_train_lbl, train_lbl))
						if val_data is not None:
							all_val_data = np.vstack((all_val_data, val_data))
							all_val_lbl = np.concatenate((all_val_lbl, val_lbl))

				data, labels = [instance], [label]
				cur_month = month
			else:
				data.append(instance)
				labels.append(label)

	return all_train_data, all_val_data, all_train_lbl, all_val_lbl, feat_order

def gen_val_indices(in_csv, k_fold):
	val_indices = {}
	cur_month, num_data = 0, 0;
	with open(in_csv, 'r') as inf:
		lines = inf.readlines()
		lines = lines[1:]
		for idx, line in enumerate(lines):
			vals = line.strip().split(" ")
			month = int(vals[0])

			if cur_month != month or idx == len(lines)-1:
				if num_data > 0:
					val_indices[cur_month] = []
					indices = np.random.permutation(num_data) 
					
					fold_size = int(num_data / k_fold)
					for i in range(k_fold):
						if i == k_fold - 1:
							start, end = i * fold_size, num_data
						else:
							start, end = i * fold_size, (i+1) * fold_size

						fold_indices = np.asarray( indices[start:end] )
						val_indices[cur_month].append(fold_indices)

				num_data = 0
				cur_month = month

			num_data += 1 
	
	data_dir = os.path.dirname(in_csv)
	for i in range(k_fold):
		fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_" + str(i))
		with open(fold_path, 'w') as outf:
			for month, indices in val_indices.items():
				outf.write(str(month) + " ")
				if k_fold > 1:
					for index in indices[i]:
						outf.write(str(index) + " ")
				outf.write("\n")
