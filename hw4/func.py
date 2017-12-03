import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding


def get_new_data(model, X_unlabeled, thrsh):
	pred = model.predict(X_unlabeled, verbose=1)
	conf_idx = (pred > thrsh) + (pred < 1 - thrsh)

	new_label = np.around(pred[conf_idx])
	new_data = np.array(X_unlabeled[conf_idx])

	conf_idx  = np.where(conf_idx)
	X_unlabeled = np.delete(X_unlabeled, conf_idx, axis=0)

	print("%d data over threshold %.2f, adding to training data" % (len(new_data), thrsh))
	return new_data, new_label


def load_wordvec(path):
	wordvec, veclen = {}, None
	with open(path, 'r') as inf:
		lines = inf.readlines()
		for idx, line in enumerate(lines):
			word, feats = line.strip().split(' ', 1)
			wordvec[word] = np.asarray(feats.strip().split(' '))
			if veclen is None:
				veclen = len(wordvec[word])

	return wordvec, veclen


def create_model(top_words, embedding_vector_length, wordvec, trainable, dropout):

	model = Sequential()
	if wordvec is not None:
		embedding_vector_length = wordvec.shape[1]
		model.add(Embedding(top_words, 
												embedding_vector_length, 
												weights=[wordvec],
												trainable=trainable,
												mask_zero=False))
	else:
		model.add(Embedding(top_words, 
												embedding_vector_length, 
												mask_zero=True))

	model.add(GRU(200, dropout= dropout, recurrent_dropout=0., return_sequences=False))
	model.add(Dense(embedding_vector_length // 2, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model


def conv_model(top_words, embedding_vector_length, wordvec, trainable, dropout):
	from keras.layers import Conv1D
	from keras.layers import MaxPooling1D

	model = Sequential()
	if wordvec is not None:
		embedding_vector_length = wordvec.shape[1]
		model.add(Embedding(top_words, 
												embedding_vector_length, 
												weights=[wordvec],
												trainable=trainable,
												mask_zero=False))
	else:
		model.add(Embedding(top_words, 
												embedding_vector_length, 
												mask_zero=False))

	model.add(Conv1D(filters=embedding_vector_length, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(GRU(100, recurrent_dropout=dropout, return_sequences=False))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model



def parse_input():
	def str2bool(v):
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')

	parser = argparse.ArgumentParser()
	parser.add_argument("--kfold", help="K-fold cross validation", type=int, default=10)
	parser.add_argument("--top_words", help="Top words in tokenization", type=int, default=10000)
	parser.add_argument("--embedding_vector_length", help="Word embedding vector size", type=int, default=64)
	parser.add_argument("--dropout", help="Dropout rate for RNN", type=float, default=0.)
	parser.add_argument("--unlabeled", help="Unlabeled data for self-learning")
	parser.add_argument("--cell", help="RNN Cell", default="GRU")
	parser.add_argument("--trainable", help="Whether embedding layer is trainable", type=str2bool, default=True)
	parser.add_argument("--thresh", help="Confidence for accepting unlabeled data", type=float, default=0.9)
	parser.add_argument("--wordvec", help="Pretrained word vector file")
	parser.add_argument("--modeldir", help="Directory for storing models")
	return parser.parse_args()
