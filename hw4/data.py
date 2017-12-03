def load_data_label(data_path, label_path, len_limit=None):
	with open(data_path, 'r') as inf:
		data = inf.readlines()

	if label_path is not None:
		with open(label_path, 'r') as inf:
			labels = [int(line.strip()) for line in inf]
	else:
		labels = None

	if len_limit is not None:
		for idx in reversed(range(len(data))):
			row = data[idx]
			length = len(row.strip().split(' '))
			if length > len_limit:
				del data[idx]
				if labels is not None:
					del labels[idx]
	
	return data, labels


def load_test_data(path):
	sequences = []
	with open(path, 'r') as inf:
		header = inf.readline()
		for line in inf:
			_, seq = line.strip().split(',', 1)
			sequences.append(seq)
	return sequences


def write_to_file(pred, path):
	with open(path, 'w') as outf:
		outf.write("id,label\n")
		for idx, res in enumerate(pred):
			outf.write(str(idx) + ',' + str(int(res[0])) + "\n")
