import os
import argparse
import keras.backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from keras.models import load_model
from splitData import *

img_rows, img_cols = 48, 48

def deprocessimage(x):
	x = (x - np.min(x)) / (np.max(x) - np.min(x))
	x = x[0, :, :, 0]
	return x

img_dir = "saliency"
if not os.path.exists(img_dir):
	os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
	os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
	os.makedirs(partial_see_dir)

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

	from scipy import misc
	img = val_data[0] * 255.0
	img = img.astype(int)
	misc.imsave(os.path.join(img_dir, 'test.png'), img[:, :, 0])

	emotion_classifier = load_model(model_path)
	print("Loaded model from {}".format(model_path))

	input_img = emotion_classifier.input
	img_ids = [0]

	for idx in img_ids:
		single_dat = np.array(val_data[idx][np.newaxis, :, :, :])
		val_proba = emotion_classifier.predict(single_dat)
		pred = val_proba.argmax(axis=-1)

		target = K.mean(emotion_classifier.output[:, pred])
		grads = K.gradients(target, input_img)[0]
		fn = K.function([input_img, K.learning_phase()], [grads])

		see = np.array(val_data[idx][:, :, 0])

		heatmap = fn([single_dat, 0])[0]
		heatmap = deprocessimage(heatmap)

		single_dat *= 255.0
		single_dat = single_dat.astype(int)
		misc.imsave(os.path.join(img_dir, 'test1.png'.format(idx)), single_dat[0, :, :, 0])

		thres = 0.5
		see[np.where(heatmap <= thres)] = np.mean(see)
		see *= 255.0
		see = see.astype(int)

		plt.figure()
		plt.imshow(heatmap, cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		test_dir = os.path.join(cmap_dir)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

		plt.figure()
		plt.imshow(see,cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		test_dir = os.path.join(partial_see_dir)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
	main()
