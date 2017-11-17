# -- coding: utf-8 --
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from splitData import *

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

img_rows, img_cols = 48, 48

def plot_confusion_matrix(cm, classes,
						  title='Confusion matrix',
						  cmap=plt.cm.jet):
	"""
	This function prints and plots the confusion matrix.
	"""
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def main():
	import sys
	model_path = sys.argv[1]

	train = "data/train.csv"
	labels, data = read_data(train, one_hot_encoding=False)

	k_fold, fold, do_zca = find_info(model_path)

	name, ext = os.path.splitext(train)
	indice_path = name + "_" + str(k_fold) + "fold"
	indices = get_val_indices(indice_path)

	_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])
	val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)

	emotion_classifier = load_model(model_path)
	np.set_printoptions(precision=2)

	predictions = emotion_classifier.predict(val_data)
	predictions = predictions.argmax(axis=-1)
	print (predictions)
	print (val_lbl)
	conf_mat = confusion_matrix(val_lbl, predictions)

	plt.figure()
	plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	plt.savefig('confusion_mat.jpg')

if __name__=='__main__':
	main()
