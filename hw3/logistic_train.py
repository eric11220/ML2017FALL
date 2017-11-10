from __future__ import print_function

import os
import sys
import random
import numpy as np
from datetime import datetime

import dataprocessing
from model import *
from splitData import *

MODLE_DIR = 'models'

img_rows, img_cols = 48, 48
nb_classes = 7
img_channels = 1

batch_size = 64

def train_logistic_model(train_data, train_lbl, val_data, val_lbl, cls, model_subdir, nb_epoch=1000):
	''' Construct model '''
	from keras.models import Sequential 
	from keras.layers import Dense, Activation 

	_, input_dim = train_data.shape
	for idx, label in enumerate(train_lbl):
		if label != cls:
			train_lbl[idx] = 0
		else:
			train_lbl[idx] = 1

	for idx, label in enumerate(val_lbl):
		if label != cls:
			val_lbl[idx] = 0
		else:
			val_lbl[idx] = 1

	model = Sequential() 
	model.add(Dense(1, input_dim=input_dim, activation='sigmoid')) 
	model.compile(optimizer='adam', 
								loss='mean_squared_error', 
								metrics=['accuracy']) 

	'''
	cls_dir = os.path.join(model_subdir, str(cls))
	if not os.path.isdir(cls_dir):
		os.mkdir(cls_dir)
	
	filepath = os.path.join(cls_dir, 'Model.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
	checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=10, save_best_only=True, mode='auto')
	'''
	
	# Only save the last model
	filepath = os.path.join(model_subdir, 'Model' + str(cls) + '_{val_loss:.4f}.hdf5')
	checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=10, save_best_only=True, mode='auto', period=nb_epoch)
	model.fit(train_data, train_lbl,
									epochs=nb_epoch,
									validation_data=(val_data, val_lbl),
									callbacks=[checkpointer])


def main():
	argc = len(sys.argv)
	if argc != 4:
		print("Usage: python3 train.py train_csv batch_size n_epoch")
		exit()

	train = sys.argv[1]
	batch_size = int(sys.argv[2])
	nb_epoch = int(sys.argv[3])

	fold_info, _ = os.path.splitext(train.split("_")[-1])
	k_fold, fold = fold_info.split("-")
	k_fold, fold = int(k_fold), int(fold)

	name, ext = os.path.splitext(train)
	indice_path = name + "_" + str(k_fold) + "fold"
	indices = get_val_indices("data/train_" + str(k_fold) + "fold")
	
	labels, data = read_data(train, one_hot_encoding=False)
	train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[fold])

	model_subdir = os.path.join(MODLE_DIR, 'feat_ensemble', os.path.splitext(os.path.basename(train))[0])
	if not os.path.isdir(model_subdir):
		os.makedirs(model_subdir)

	for cls in range(nb_classes):
		train_logistic_model(train_data, np.array(train_lbl), val_data, np.array(val_lbl), cls, model_subdir, nb_epoch)
		
if __name__ == '__main__':
	main()
