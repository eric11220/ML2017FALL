import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import Callback


class RandomSaver(Callback):
	def __init__(self, images, modeldir, num=40, per_row=10):
		self.images = images
		self.modeldir = modeldir
		self.num = num
		self.per_row = per_row

	def on_epoch_end(self, epoch, log={}):
		images, modeldir, num, per_row = self.images, self.modeldir, self.num, self.per_row
		num_rows = math.ceil(self.num * 2 / per_row)

		idx = np.random.choice(len(images), num)
		orig = images[idx]
		reconst = np.reshape(self.model.predict(orig), (-1, 28, 28))
		orig = np.reshape(images[idx], (-1, 28, 28))
		
		fig = plt.figure()
		for idx, (orig_im, reconst_im) in enumerate(zip(orig, reconst)):
			fig.add_subplot(num_rows, per_row, 2*idx+1)
			plt.imshow(orig_im, cmap='gray')

			fig.add_subplot(num_rows, 10, 2*idx+2)
			plt.imshow(reconst_im, cmap='gray')

		img_dir = os.path.join(modeldir , 'images')
		if not os.path.isdir(img_dir):
			os.makedirs(img_dir)

		img_path = os.path.join(img_dir, "epoch" + str(epoch) + ".jpg")
		plt.savefig(img_path)
		plt.close()


def auto_encoder(shape, dropout=0.):

	inputs = Input(shape=(shape,))

	encoded = Dense(128, activation="relu")(inputs)
	encoded = Dense(64, activation="relu")(encoded)
	encoded = Dense(32, activation="relu")(encoded)

	decoded = Dense(64, activation="relu")(encoded)
	decoded = Dense(128, activation="relu")(decoded)
	x = Dense(shape, activation="sigmoid")(decoded)

	autoencoder = Model(inputs=inputs, outputs=x)
	encoder = Model(inputs=inputs, outputs=encoded)

	autoencoder.compile('adam', 'mean_squared_error')
	autoencoder.summary()
	return autoencoder, encoder


def conv_auto_encoder(shape):
	from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

	inputs = Input(shape=shape)
	encoded = Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
	encoded = MaxPooling2D((2,2), padding="same")(encoded)
	encoded = Conv2D(8, (3,3), activation="relu", padding="same")(encoded)
	encoded = MaxPooling2D((2,2), padding="same")(encoded)
	encoded = Conv2D(8, (3,3), activation="relu", padding="same")(encoded)
	encoded = MaxPooling2D((2,2), padding="same")(encoded)

	decoded = Conv2D(8, (3,3), activation="relu", padding="same")(encoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(8, (3,3), activation="relu", padding="same")(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(16, (3,3), activation="relu")(decoded)
	decoded = UpSampling2D((2,2))(decoded)
	decoded = Conv2D(1, (3,3), activation="sigmoid", padding="same")(decoded)

	autoencoder = Model(inputs=inputs, outputs=decoded)
	encoder = Model(inputs=inputs, outputs=encoded)

	autoencoder.compile('adam', 'mean_squared_error')
	autoencoder.summary()
	return autoencoder, encoder
