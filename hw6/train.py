import os
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras.models import load_model

from model import *
from func import *


MODEL_DIR='models'
nb_epoch = 50


def main():
	images = load_img()
	images = images / 255.0


	time_now = datetime.now().strftime('%d-%H-%M')
	modeldir = os.path.join(MODEL_DIR, time_now)
	os.makedirs(modeldir, exist_ok=True)

	model, encoder = auto_encoder(images.shape[1])
	model.fit(images, 
						images,
						epochs=nb_epoch,
						callbacks=[RandomSaver(images, modeldir)])

	encoder.save(os.path.join(modeldir, 'encoder.h5'))
	model.save(os.path.join(modeldir, 'model.h5'))

	encoded = encoder.predict(images)
	encoded = np.reshape(encoded, (encoded.shape[0], -1))
	kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded)
	np.save(os.path.join(modeldir, "kmeans.npy"), kmeans.cluster_centers_)


	questions = load_question()
	answers = answer_question(questions, kmeans.labels_)
	write_ans_to_file(answers)


if __name__ == '__main__':
	main()
