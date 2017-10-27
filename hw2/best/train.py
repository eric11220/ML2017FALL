import os
import sys
import math
import json
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

from splitData import *

np.set_printoptions(suppress=True)


NN_MODEL_DIR = 'models'
SVC_MODEL_DIR = 'svc_models'
GBDT_MODEL_DIR = 'gbdt_models'

num_losses = 10
tolerance = 0.001
epsilon = 1e-8

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


def nn_train(train_data, train_label, val_data, val_label, config, n_epoch=100, lr=1, batch_size=1, display_epoch=10, lamb=0.1, early_stop=False, patience=5):
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Dropout
	from keras.callbacks import EarlyStopping

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


def save_svc_model(model, k_fold, fold_idx, train_mean, train_std, lamb):
	from sklearn.externals import joblib

	model_path = "svc_" + "C=" + str(lamb) + "_" + str(k_fold) + '-' + str(fold_idx)
	model_path = os.path.join(SVC_MODEL_DIR, model_path)
	mean_std_path = model_path + '_mean_std.npy'

	# Save svc model and train mean and std
	joblib.dump(model, model_path, compress=9)
	np.save(mean_std_path, [train_mean, train_std])


def svc_train(train_data, train_label, val_data, val_label, k_fold, fold_idx, train_mean, train_std, lamb=0.1, force_rebuild=False):
	from sklearn.svm import SVC
	from sklearn.externals import joblib

	train_label = np.reshape(train_label, [-1])
	num_data, dim = train_data.shape

	model_path = "svc_" + str(k_fold) + '-' + str(fold_idx)
	model_path = os.path.join(SVC_MODEL_DIR, model_path)
	if os.path.isfile(model_path) and force_rebuild is False:
		clf = joblib.load(model_path)
	else:
		clf = SVC(C=lamb)
		clf.fit(train_data, train_label)


	train_pred = clf.predict(train_data)
	#print("Training Accuracy:" + str(np.sum(train_pred == train_label) / len(train_label)))

	eval_acc = 0
	if val_data is not None:
		val_pred = clf.predict(val_data)
		val_label = np.reshape(val_label, [-1])
		eval_acc = np.sum(val_pred == val_label) / len(val_label)
		#print("Testing Accuracy:" + str(eval_acc))

	return clf, (0., eval_acc)


def gbdt_train(train_data, train_label, val_data, val_label, tree_depth=3, num_round=400):

	# Convert np array to DMatrix
	train_label = np.reshape(train_label, [-1])
	train_data_lbl = xgb.DMatrix(train_data, label=train_label)

	if val_data is not None:
		val_label = np.reshape(val_label, [-1])
		val_data_lbl = xgb.DMatrix(val_data, label=val_label)


	param = {'max_depth': tree_depth, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8}
	param['nthread'] = 4
	param['eval_metric'] = 'error'

	evallist = [(train_data_lbl, 'train')]
	if val_data is not None:
		evallist.append((val_data_lbl, 'eval'))

	bst = xgb.train(param, train_data_lbl, num_round, evallist, early_stopping_rounds=20)
	ypred = bst.predict(val_data_lbl)

	eval_acc = None
	if val_data is not None:
		eval_acc = np.sum(np.around(ypred) == val_label) / len(val_label)

	return bst, (0., eval_acc)


def gbdt_save_model(model, k_fold, fold_idx, train_mean, train_std):

	model_path = str(k_fold) + '-' + str(fold_idx)
	model_path = os.path.join(GBDT_MODEL_DIR, model_path)
	mean_std_path = model_path + '_mean_std.npy'

	# Save gbdt model and train mean and std
	model.save_model(model_path)
	np.save(mean_std_path, [train_mean, train_std])


def main():
	argc = len(sys.argv)
	if argc != 10:
		print("Usage: python train.py X_train Y_train k_fold n_epoch lambda lr batch_size nn_config classifier")
		exit()

	X_train = sys.argv[1]
	Y_train = sys.argv[2]
	k_fold = int(sys.argv[3])
	n_epoch = int(sys.argv[4])
	lamb = float(sys.argv[5])
	lr = float(sys.argv[6])
	batch_size = int(sys.argv[7])
	config_path = sys.argv[8]
	classifier = sys.argv[9]

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

	numeric_indices, data = read_data(X_train)
	if classifier == "nn": # or classifier == "gbdt":
		labels = read_label(Y_train)
	else:
		_, labels = read_data(Y_train, dtype=np.int16)

	print(numeric_indices)
	if numeric_indices[0] != '':
		numeric_indices = np.asarray(numeric_indices, dtype=np.uint8)


	sum_acc, sum_err = 0, 0
	for i in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[i])

		train_mean = np.zeros((train_data.shape[1], ))
		train_std = np.ones((train_data.shape[1], ))
		if numeric_indices[0] != '':
			train_mean[numeric_indices] = np.mean(train_data, axis=0)[numeric_indices]
			train_std[numeric_indices] = np.std(train_data, axis=0)[numeric_indices]

		'''
		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		'''

		train_data = (train_data - train_mean) / (train_std + epsilon)
		if val_data is not None:
			val_data = (val_data - train_mean) / (train_std + epsilon)

		if classifier == "nn":
			model, stat = nn_train(train_data, train_lbl, val_data, val_lbl, config, n_epoch=n_epoch, batch_size=batch_size, lamb=lamb, early_stop=False)
			save_nn_model(model, config, k_fold, i, train_mean, train_std)
		elif classifier == "svc":
			model, stat = svc_train(train_data, train_lbl, val_data, val_lbl, k_fold, i, train_mean, train_std, lamb=lamb)
			save_svc_model(model, k_fold, i, train_mean, train_std, lamb)
		elif classifier == "gbdt":
			model, stat = gbdt_train(train_data, train_lbl, val_data, val_lbl)
			gbdt_save_model(model, k_fold, i, train_mean, train_std)

		if stat is not None:
			sum_acc += stat[1]
			sum_err += stat[0]
			print("fold %d, acc: %.4f" % (i, stat[1]))

	if  k_fold > 1:
		sys.stderr.write("Accuracy:" + str(sum_acc / k_fold) + "\n")
		sys.stderr.write("Error:" + str(sum_err / k_fold) + "\n")


if __name__ == '__main__':
	main()
