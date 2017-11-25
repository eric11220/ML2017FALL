import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from keras.models import load_model
from keras import backend as K
from splitData import *

img_rows, img_cols = 48, 48

vis_dir = os.path.join('image','vis_layer')
if not os.path.exists(vis_dir):
	os.makedirs(vis_dir)
filter_dir = os.path.join('image','vis_filter')
if not os.path.exists(filter_dir):
	os.makedirs(filter_dir)

nb_class = 7
LR_RATE = 2 * 1e-2
NUM_STEPS = 200
RECORD_FREQ = 10

def deprocess_image(x):
	"""
	As same as that in problem 4.
	"""
	return x

def main():
	parser = argparse.ArgumentParser(prog='filter_analyze.py', description='Visualize CNN filter.')
	parser.add_argument('--model',type=str,metavar='<modelPath>',required=True)
	parser.add_argument('--mode',type=int,metavar='<visMode>',default=1,choices=[1,2])
	args = parser.parse_args()

	model_path = args.model
	train = "data/train.csv"
	labels, data = read_data(train, one_hot_encoding=False)

	k_fold, fold, do_zca = find_info(model_path)

	name, ext = os.path.splitext(train)
	indice_path = name + "_" + str(k_fold) + "fold"
	indices = get_val_indices(indice_path)

	_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])
	val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)

	emotion_classifier = load_model(model_path)

	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

	def normalize(x):
		# utility function to normalize a tensor by its L2 norm
		return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

	def grad_ascent(filter_imgs, input_image_data,iter_func):
		for step in range(NUM_STEPS):
			loss, grad = iter_func([input_image_data, 0])
			input_image_data += grad
			'''
			print(loss)
			print(grad)
			print(grad.shape)
			input("")
			'''
			if step % RECORD_FREQ == 0:
				filter_imgs[step // RECORD_FREQ].append((input_image_data, loss))

	input_img = emotion_classifier.input
	# visualize the area CNN see
	if args.mode == 1:
		collect_layers = list()
		collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['block2_pool'].output]))

		choose_id = 0
		photo = val_data[choose_id]

		for cnt, fn in enumerate(collect_layers):
			im = fn([photo.reshape(1, img_rows, img_cols,1),0])
			fig = plt.figure(figsize=(14,8))
			nb_filter = im[0].shape[3]

			for i in range(nb_filter):
				print("processing filter %d..." % i)
				ax = fig.add_subplot(nb_filter/16,16,i+1)
				ax.imshow(im[0][0,:,:,i],cmap='gray')
				plt.xticks(np.array([]))
				plt.yticks(np.array([]))
				plt.tight_layout()

			fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
			fig.savefig(os.path.join(vis_dir,'layer{}'.format(cnt)))

	else:
		name_ls = ['block4_pool']
		collect_layers = list()
		collect_layers.append(layer_dict[name_ls[0]].output)

		for cnt, c in enumerate(collect_layers):
			filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
			nb_filter = c.shape[-1]

			nb_filter = min(nb_filter, 64)
			for filter_idx in range(nb_filter):
				print("Processing filter %d..." % filter_idx)
				input_img_data = np.random.random((1, img_rows, img_cols, 1))
				loss = K.mean(c[:,:,:,filter_idx])
				grads = normalize(K.gradients(loss,input_img)[0])
				iterate = K.function([input_img, K.learning_phase()],[loss,grads])

				grad_ascent(filter_imgs, input_img_data, iterate)

			fig = plt.figure(figsize=(14,8))
			for i in range(nb_filter):
				ax = fig.add_subplot(int(nb_filter)/16,16,i+1)
				ax.imshow(filter_imgs[-1][i][0][0, :, :, 0],cmap='gray')
				plt.xticks(np.array([]))
				plt.yticks(np.array([]))
				plt.xlabel('filter %d' % i)
				plt.tight_layout()
			fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0], NUM_STEPS))

			img_path = os.path.join(filter_dir,'{}'.format(name_ls[0]))
			if not os.path.isdir(img_path):
				os.mkdir(img_path)
			fig.savefig(os.path.join(img_path,'e{}'.format(NUM_STEPS)))

if __name__ == "__main__":
	main()
