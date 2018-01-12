import os
import sys
from func import *
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def distance(pt, center):
	return np.sqrt(np.sum(np.power(pt - center, 2)))


def main():
	encoder_path, img_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

	# load images
	imgs = np.load(img_path)
	imgs = imgs.astype(np.float32)
	imgs /= 255.0
	
	encoder = load_model(encoder_path)
	encoded = encoder.predict(imgs)

	modeldir = os.path.dirname(encoder_path)
	centers = np.load(os.path.join(modeldir, "kmeans.npy"))

	labels = []
	for img in encoded:
		dists = []
		for center in centers:
			dists.append(distance(img, center))
		labels.append(np.argmax(dists))
	labels = np.asarray(labels)

	questions = load_question(path=test_path)
	answers = answer_question(questions, labels)
	write_ans_to_file(answers, path=out_path)


if __name__ == '__main__':
	main()
