import os
import sys
import math
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from splitData import *

np.set_printoptions(suppress=True)

MODEL_DIR = 'models'

num_losses = 10
tolerance = 0.001
epsilon = 1e-8


def save_model(model, layers, k_fold, fold_idx):
	model_path = os.path.join(MODEL_DIR, "_".join([str(l) for l in layers]) + '_' + str(k_fold) + '-' + fold_idx)
	weight_path = os.path.join(MODEL_DIR, "_".join([str(l) for l in layers]) + '_' + str(k_fold) + '-' + fold_idx + '_weight.h5')
	print(model_path, weight_path)
	input("")

	model_json = model.to_json()
	with open(model_path, "w") as outf:
		outf.write(model_json)
	model.save_weight(weight_path)


def train(train_data, train_label, val_data, val_label, config, n_epoch=100, lr=1, batch_size=1, display_epoch=10, lamb=0.1, early_stop=False):
	num_data, dim = train_data.shape

	layers, do_dropout, drop_rate = config["nn_layers"], config["dropout"], config["drop_rate"]

	model = Sequential()
	model.add(Dense(input_dim=dim, output_dim=layers[0]))
	model.add(Activation('relu'))
	if do_dropout[0] is True:
		model.add(Dropout(drop_rate))

	for layer, dropout in zip(layers[1:], do_dropout[1:]):
		model.add(Dense(output_dim=layer))
		model.add(Activation('relu'))
		if dropout is True:
			model.add(Dropout(drop_rate))

	model.add(Dense(output_dim=2))
	model.add(Activation('softmax'))

	model.compile(	loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=n_epoch, validation_data=(val_data, val_label))
	return model


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

		model = train(train_data, train_lbl, val_data, val_lbl, config, n_epoch=n_epoch, batch_size=batch_size, lamb=lamb, early_stop=False)
		save_model(model, config["nn_layers"], k_fold, i)
		input("")

		if acc is not None:
			sum_acc += acc

	if  k_fold > 1:
		sys.stderr.write(str(sum_error / k_fold) + "\n")


if __name__ == '__main__':
	main()
