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

def gbdt_modelfit(train_data, train_label, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):

	'''
	# Step 1
	alg = XGBClassifier(
		learning_rate=0.1,
		n_estimators=1000,
		max_depth=5,
		min_child_weight=1,
		gamma=0,
		subsample=0.8,
		colsample_bytree=0.8,
		objective= 'binary:logistic',
		nthread=4,
		scale_pos_weight=1,
		seed=27)

	# Convert np array to DMatrix
	train_label = np.reshape(train_label, [-1])
	train_data_lbl = xgb.DMatrix(train_data, label=train_label)
	
	if useTrainCV is True:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(train_data, train_label)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
			metrics='error', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
		alg.set_params(n_estimators=cvresult.shape[0])
	
	# Fit the algorithm on the data
	alg.fit(train_data, train_label, eval_metric='auc')
		
	# Predict training set:
	dtrain_predictions = alg.predict(train_data)
	dtrain_predprob = alg.predict_proba(train_data)[:,1]
		
	# Print model report:
	print("\nModel Report")
	print("Accuracy : %.4g" % metrics.accuracy_score(train_label, dtrain_predictions))
	'''

	# Step 2
	param_test1 = {
	 'max_depth':range(3,10,2),
	 'min_child_weight':range(1,6,2)
	}
	gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=137, max_depth=5,
	min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
	param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	gsearch1.fit(train[predictors],train[target])
	gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


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
	if classifier == "nn":
		labels = read_label(Y_train)
	else:
		_, labels = read_data(Y_train, dtype=np.int16)
	numeric_indices = np.asarray(numeric_indices, dtype=np.uint8)


	gbdt_modelfit(data, labels, useTrainCV=True, cv_folds=5, early_stopping_rounds=50)
	input("")


	sum_acc, sum_err = 0, 0
	for i in range(k_fold):
		train_data, val_data, train_lbl, val_lbl = split_train_val(data, labels, indices[i])

		train_mean = np.zeros((train_data.shape[1], ))
		train_std = np.ones((train_data.shape[1], ))
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
			input("")

	if  k_fold > 1:
		sys.stderr.write("Accuracy:" + str(sum_acc / k_fold) + "\n")
		sys.stderr.write("Error:" + str(sum_err / k_fold) + "\n")


if __name__ == '__main__':
	main()
