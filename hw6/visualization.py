import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.models import load_model


def distance(pt, center):
	return np.sqrt(np.sum(np.power(pt - center, 2)))


def main():
	# load images
	imgs = np.load("data/visualization.npy")
	imgs = imgs.astype(np.float32)

	imgs /= 255.0
	print(imgs.shape)
	
	encoder = load_model("models/19-20-28/encoder.h5")
	encoded = encoder.predict(imgs)
	centers = np.load("models/19-20-28/kmeans.npy")

	labels = []
	for img in encoded:
		dists = []
		for center in centers:
			dists.append(distance(img, center))
		labels.append(np.argmax(dists))
	labels = np.asarray(labels)
	labelzero = labels == 0
	labelone = labels == 1

	if not os.path.isfile("tsne.npy"):
		encoded = TSNE(n_components=2).fit_transform(encoded)
		np.save("tsne.npy", encoded)
	else:
		encoded = np.load("tsne.npy")

	plt.clf()
	plt.scatter(encoded[labelzero, 0], encoded[labelzero, 1], c='b', label='dataset A', s=0.2)
	plt.scatter(encoded[labelone, 0], encoded[labelone, 1], c='r', label='dataset B', s=0.2)
	plt.legend()
	plt.savefig('predict.png')

	plt.clf()
	plt.scatter(encoded[:5000, 0], encoded[:5000, 1], c='b', label='dataset A', s=0.2)
	plt.scatter(encoded[5000:, 0], encoded[5000:, 1], c='r', label='dataset B', s=0.2)
	plt.legend()
	plt.savefig('correct.png')


if __name__ == '__main__':
	main()
