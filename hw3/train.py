# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train.py
from __future__ import print_function

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

from datetime import datetime
from dataprocessing import *
from model import *
from splitData import *

MODLE_DIR = 'models'

def main():
	argc = len(sys.argv)
	if argc != 8:
		print("Usage: python train.py k_fold batch_size n_epoch model_struct loss dropout do_zca")
		exit()

	img_rows, img_cols = 48, 48
	k_fold = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	nb_epoch = int(sys.argv[3])
	model_struct = sys.argv[4]
	loss = sys.argv[5]
	dropout = float(sys.argv[6])
	do_zca = sys.argv[7] == "1"

	nb_classes = 7
	img_channels = 1
	train = "data/train.csv"

	from keras.utils import np_utils
	from keras.preprocessing.image import ImageDataGenerator

	if k_fold > 1:
		name, ext = os.path.splitext(train)
		indice_path = name + "_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(train, k_fold)
		else:
			indices = get_val_indices(indice_path)
	else:
		indices = [[]]
	
	labels, data = read_data(train, one_hot_encoding=True)
	
	time_now = datetime.now().strftime('%m-%d_%H:%M')
	fold_order = list(range(k_fold))
	fold_order = [0]

	sum_acc, sum_err = 0, 0
	for fold in fold_order:

		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[fold])

		if do_zca is True:
			model_subdir = os.path.join(MODLE_DIR, model_struct, "zca_" + time_now, str(k_fold) + "-" + str(fold))
			os.makedirs(model_subdir)

			train_data, zca_mat, train_mean = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(train_data)
			val_data, _, _ = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(val_data, zca_mat, train_mean)
			zca_path = os.path.join(model_subdir, 'zca_matrix.npy')
			np.savez(zca_path, zca_mat, train_mean)
		else:
			model_subdir = os.path.join(MODLE_DIR, model_struct, time_now, str(k_fold) + "-" + str(fold))
			os.makedirs(model_subdir)
	
		train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols)
		train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
		val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols)
		val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)
		
		if model_struct == "orig":
			model = orig_model(loss)
		elif model_struct == "vgg16":
			model = vgg16(loss, dropout)
		elif model_struct == "fc":
			model = fc_model(loss)
			train_data = train_data.reshape(train_data.shape[0], -1)
			val_data = val_data.reshape(val_data.shape[0], -1)
			print(train_data.shape)
			print(val_data.shape)
		elif model_struct == "resnet50":
			from scipy import misc
			train_data = np.asarray([misc.imresize(np.reshape(img, [img_rows, img_cols]), (224, 224)) for img in train_data])
			val_data	= np.asarray([misc.imresize(np.reshape(img, [img_rows, img_cols]), (224, 224)) for img in val_data])
			train_data = train_data[:, :, :, np.newaxis]
			val_data = val_data[:, :, :, np.newaxis]
			model = resnet50()
		
		filepath = os.path.join(model_subdir, 'Model.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
		checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=10, save_best_only=True, mode='auto')
		tensorboard = keras.callbacks.TensorBoard()
		
		if model_struct != 'fc':
			datagen = ImageDataGenerator(
				zoom_range=[0.9, 1.1],
				shear_range=20,
				rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True)  # randomly flip images
					#vertical_flip=False)  # randomly flip images

			datagen.fit(train_data)
			hist = model.fit_generator(datagen.flow(train_data, train_lbl,
								batch_size=batch_size),
								train_data.shape[0]/batch_size,
								epochs=nb_epoch,
								validation_data=(val_data, val_lbl),
								callbacks=[checkpointer])
		else:
			hist = model.fit(train_data,
											train_lbl,
											batch_size=batch_size,
											epochs=nb_epoch,
											validation_data=(val_data, val_lbl),
											callbacks=[checkpointer])

		# Plot training and validation accuracies
		val_acc = hist.history["val_acc"]
		train_acc = hist.history["acc"]

		from matplotlib import pyplot as plt

		plt.title("Accuracy of model")
		plt.plot(train_acc, color='b', label='train')
		plt.plot(val_acc, color='r', label='valid')
		plt.legend(loc='lower right')

		plt.ylabel("Accuracy")
		plt.xlabel("# of epoch")

		plt.tight_layout()
		plt.savefig(os.path.join(model_subdir, "train_val_acc.jpg"))
		
if __name__ == '__main__':
	main()
