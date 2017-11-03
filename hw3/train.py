# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 train.py
from __future__ import print_function

import os
import sys
import numpy as np
from datetime import datetime

import dataprocessing
from model import *
from splitData import *

MODLE_DIR = 'models'


def main():
	argc = len(sys.argv)
	if argc != 5:
		print("Usage: python train.py k_fold batch_size n_epoch model_struct")

	img_rows, img_cols = 48, 48
	k_fold = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	nb_epoch = int(sys.argv[3])
	model_struct = sys.argv[4]

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
	
	labels, data = read_data(train, one_hot_encoding=False)
	
	sum_acc, sum_err = 0, 0
	for fold in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[fold])
		Train_x, Train_y = dataprocessing.convert_data(train_data, train_lbl)
		Val_x, Val_y = dataprocessing.convert_data(val_data, val_lbl)
	
		Train_x = np.asarray(Train_x) 
		Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols)
		Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols, 1)
		Train_x = Train_x.astype('float32')
		
		Val_x = np.asarray(Val_x)
		Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols)
		Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols, 1)
		Val_x = Val_x.astype('float32')
		
		Train_y = np_utils.to_categorical(Train_y, nb_classes)
		Val_y = np_utils.to_categorical(Val_y, nb_classes)
		
		if model_struct == "orig":
			model = orig_model()
		elif model_struct == "vgg16":
			model = vgg16()
		elif model_struct == "resnet50":
			from scipy import misc
			Train_x = np.asarray([misc.imresize(np.reshape(img, [img_rows, img_cols]), (224, 224)) for img in Train_x])
			Val_x	= np.asarray([misc.imresize(np.reshape(img, [img_rows, img_cols]), (224, 224)) for img in Val_x])
			Train_x = Train_x[:, :, :, np.newaxis]
			Val_x = Val_x[:, :, :, np.newaxis]

			model = resnet50()
		
		model_subdir = os.path.join(MODLE_DIR, model_struct, datetime.now().strftime('%m-%d_%H:%M'), str(k_fold) + "-" + str(fold))
		os.makedirs(model_subdir)

		filepath = os.path.join(model_subdir, 'Model.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
		checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=10, save_best_only=True, mode='auto')
		
		datagen = ImageDataGenerator(
		    featurewise_center=False,  # set input mean to 0 over the dataset
		    samplewise_center=False,  # set each sample mean to 0
		    featurewise_std_normalization=False,  # divide inputs by std of the dataset
		    samplewise_std_normalization=False,  # divide each input by its std
		    zca_whitening=False,  # apply ZCA whitening
		    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
		    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
		    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
		    horizontal_flip=True,  # randomly flip images
		    vertical_flip=False)  # randomly flip images
		
		datagen.fit(Train_x)
		model.fit_generator(datagen.flow(Train_x, Train_y,
		                    batch_size=batch_size),
		                    Train_x.shape[0]/batch_size,
		                    epochs=nb_epoch,
		                    validation_data=(Val_x, Val_y),
		                    callbacks=[checkpointer])
		
if __name__ == '__main__':
	main()
