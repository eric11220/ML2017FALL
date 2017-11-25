import os
import sys
from splitData import *

def main():
	argc = len(sys.argv)
	if argc != 4:
		print("usage: python train.py training_txt training_label fold")
		exit()

	data_path = sys.argv[1]
	label_path = sys.argv[2]
	k_fold = int(sys.argv[3])

	data_dir = os.path.dirname(data_path)

	name, ext = os.path.splitext(data_path)
	indice_path = name + "_" + str(k_fold) + "fold"
	if not os.path.isfile(indice_path):
		indices = gen_val_indices(data_path, k_fold, indice_path)
	else:
		indices = get_val_indices(indice_path)

	with open(data_path, 'r') as inf:
		data = np.asarray(inf.readlines())
	with open(label_path, 'r') as inf:
		labels = np.asarray(inf.readlines())


	# perform k-fold cross validation
	if k_fold > 1:
		for fold in range(k_fold):
			print("Processing fold %d..."  % fold)
			train_data, val_data, train_label, val_label = split_train_val(data, labels, indices[fold])

			cv_dir = os.path.join(data_dir, 'cv' + str(fold))
			if not os.path.isdir(cv_dir):
				os.makedirs(cv_dir)


			train_data_path = os.path.join(cv_dir, 'train_data.txt')
			with open(train_data_path, 'w') as outf:
				for row in train_data:
					outf.write(row)
			print("Writing training data...")

			train_label_path = os.path.join(cv_dir, 'train_label.txt')
			with open(train_label_path, 'w') as outf:
				for row in train_label:
					outf.write(row)
			print("Writing training label...")

			val_data_path = os.path.join(cv_dir, 'val_data.txt')
			with open(val_data_path, 'w') as outf:
				for row in val_data:
					outf.write(row)
			print("Writing validation data...")

			val_label_path = os.path.join(cv_dir, 'val_label.txt')
			with open(val_label_path, 'w') as outf:
				for row in val_label:
					outf.write(row)
			print("Writing validation label...")


if __name__ == '__main__':
	main()
