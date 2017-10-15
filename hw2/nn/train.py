import os
import sys
import math
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from splitData import *

np.set_printoptions(suppress=True)

COEF_DIR = 'coefficients'

num_losses = 10
tolerance = 0.001
epsilon = 1e-8
drop_rate = 0.5


def train(train_data, train_label, val_data, val_label, layers, n_epoch=100, lr=1, batch_size=1, display_epoch=10, lamb=0.1, early_stop=False):
	num_data, dim = train_data.shape

	model = Sequential()
	model.add(Dense(input_dim=dim, output_dim=layers[0]))
	model.add(Activation('relu'))
	model.add(Dropout(drop_rate))

	for layer in layers[1:]:
		model.add(Dense(output_dim=layer))
		model.add(Activation('relu'))
		model.add(Dropout(drop_rate))

	model.add(Dense(output_dim=2))
	model.add(Activation('softmax'))

	model.compile(	loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=n_epoch, validation_data=(val_data, val_label))
	train_acc = model.evaluate(train_data, train_label)
	val_acc = model.evaluate(val_data, val_label)


def main():
	argc = len(sys.argv)
	if argc != 9:
		print("Usage: python train.py X_train Y_train k_fold n_epoch lambda lr batch_size nn_config")
		exit()

	X_train = sys.argv[1]
	Y_train = sys.argv[2]
	k_fold = int(sys.argv[3])
	n_epoch = int(sys.argv[4])
	lamb = float(sys.argv[5])
	lr = float(sys.argv[6])
	batch_size = int(sys.argv[7])
	config_path = sys.argv[8]

	with open(config_path, "r") as inf:
   		config = json.load(inf)	

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
	labels = read_label(Y_train, n_class=2)

	sum_error = 0
	for i in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[i])

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		train_data = (train_data - train_mean) / train_std

		if val_data is not None:
			val_data = (val_data - train_mean) / train_std

		coef, acc = train(train_data, train_lbl, val_data, val_lbl, config["nn_layers"], n_epoch=n_epoch, batch_size=batch_size, lamb=lamb, early_stop=False)

		if acc is not None:
			sum_acc += acc

	if  k_fold > 1:
		sys.stderr.write(str(sum_error / k_fold) + "\n")


if __name__ == '__main__':
	main()
