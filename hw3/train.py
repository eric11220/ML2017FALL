import os
import sys
import math
import json
import numpy as np

from splitData import *

np.set_printoptions(suppress=True)
NN_MODEL_DIR = 'models'

num_losses = 10
tolerance = 0.001
epsilon = 0

def save_nn_model(model, config, k_fold, fold_idx, train_mean, train_std):
	layers, do_dropout, drop_rate = config["nn_layers"], config["dropout"], config["drop_rate"]

	model_path = ""
	for l, d in zip(layers, do_dropout):
		model_path += str(l)
		if d is True:
			model_path += "d" + str(drop_rate)
		model_path += "_"

	model_path += str(k_fold) + '-' + str(fold_idx)
	model_path = os.path.join(NN_MODEL_DIR, model_path)
	weight_path = model_path + '_weight.h5'
	mean_std_path = model_path + '_mean_std.npy'

	model_json = model.to_json()
	with open(model_path, "w") as outf:
		outf.write(model_json)

	model.save_weights(weight_path)
	np.save(mean_std_path, [train_mean, train_std])


def construct_model(config):
	from keras.models import Sequential
	from keras.layers import Conv2D, Dense, Activation, Dropout, MaxPooling2D, Flatten

	model, flattened = Sequential(), False
	for layer_id, layer in enumerate(config["layers"]):
		form, attr = list(layer.items())[0]
		ksize, stride, channel, drop_rate = attr.get("ksize", None), attr.get("stride", None), attr.get("channel", None), 1-attr.get("dropout", 0)
		print(ksize, stride, channel, drop_rate)

		if form == "conv":
			if  layer_id == 0:
				model.add(Conv2D(channel, 
								 ksize, 
								 strides=(stride, stride), 
								 input_shape=(48, 48, 1),
								 padding="same", data_format="channels_last", activation="relu", kernel_initializer='glorot_uniform') )
			else:
				model.add(Conv2D(channel, 
								 ksize, 
								 strides=(stride, stride), 
								 padding="same", data_format="channels_last", activation="relu", kernel_initializer='glorot_uniform') )
		elif form == "pool":
			model.add(MaxPooling2D( pool_size=(ksize, ksize),
									padding="same", data_format="channels_last"))
		elif form == "fc":
			if flattened is False:
				model.add(Flatten())
				flattened = True
			model.add(Dense(units=channel))
			model.add(Dropout(drop_rate))

	model.add(Dense(units=6))
	model.add(Activation('softmax'))
	model.compile(	loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	#print(model.summary())
	return model


def nn_train(train_data, train_label, val_data, val_label, config, n_epoch=100, lr=1, batch_size=1, display_epoch=10, lamb=0.1, early_stop=False, patience=30):
	from keras.callbacks import EarlyStopping

	num_data = train_data.shape[0]
	print("Number of Training data: %d" % num_data)

	model = construct_model(config)
	early_stop = EarlyStopping(	monitor='val_loss',
								min_delta=0,
								patience=patience,
								verbose=0)

	if val_data is not None:
		model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=n_epoch, callbacks=[early_stop] ,validation_data=(val_data, val_label))
		return model, model.evaluate(val_data, val_label)
	else:
		model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=n_epoch)
		return model, None


# Subtract mean per image and then setting norm to 100
def champion_norm(data):
	image_mean = np.mean(data, axis=1)
	data = data - image_mean

	norm1 = np.sum(np.abs(data))
	data *= 100/norm1
	return data


>>>>>>> 8177aeb130a00a23cd28f969c1dce31b58293563
def main():
	argc = len(sys.argv)
	if argc != 8:
		print("Usage: python train.py train k_fold n_epoch lambda lr batch_size nn_config")
		exit()

	train = sys.argv[1]
	k_fold = int(sys.argv[2])
	n_epoch = int(sys.argv[3])
	lamb = float(sys.argv[4])
	lr = float(sys.argv[5])
	batch_size = int(sys.argv[6])
	config_path = sys.argv[7]

	with open(config_path, "r") as inf:
   		config = json.load(inf)	

	if k_fold > 1:
		name, ext = os.path.splitext(train)
		indice_path = name + "_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(train, k_fold)
		else:
			indices = get_val_indices(indice_path)
	else:
		indices = [[]]

	labels, data = read_data(train)
	data = np.reshape(data, [-1, 48, 48, 1])
	
	sum_acc, sum_err = 0, 0
	for i in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[i])

		train_data = champion_norm(train_data)

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)

		train_data = (train_data - train_mean) / (train_std + epsilon)
		if val_data is not None:
			val_data = (val_data - train_mean) / (train_std + epsilon)

		model, stat = nn_train(train_data, train_lbl, val_data, val_lbl, config, n_epoch=n_epoch, batch_size=batch_size, lamb=lamb, early_stop=False)
		save_nn_model(model, config, k_fold, i, train_mean, train_std)

		if stat is not None:
			sum_acc += stat[1]
			sum_err += stat[0]
			print("fold %d, acc: %.4f" % (i, stat[1]))

	if  k_fold > 1:
		sys.stderr.write("Accuracy:" + str(sum_acc / k_fold) + "\n")
		sys.stderr.write("Error:" + str(sum_err / k_fold) + "\n")


if __name__ == '__main__':
	main()
