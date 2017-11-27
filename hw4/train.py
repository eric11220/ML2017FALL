# python3 train.py data/processed/ 5

import sys
import keras
import time
import shutil
import argparse
from datetime import datetime

from splitData import *
from index_words import *

nb_classes = 2
batch_size = 32
nb_epochs = 10
MODEL_DIR = 'models'

def load_wordvec(path):
	wordvec, word_index = [], {}
	with open(path, 'r') as inf:
		lines = inf.readlines()
		for idx, line in enumerate(lines):
			line = line.strip().split(' ')
			word, vector = line[0], line[1:]
			wordvec.append(vector)
			if idx != len(lines)-1:
				word_index[word] = idx+1

	return np.asarray(wordvec, dtype=np.float32), word_index


def gru(vocab_size, wordvec_len=256, dropout=0.2, wordvec=None):
	from keras.models import Sequential
	from keras.layers import Embedding, GRU, Dense, Dropout
	from keras.optimizers import Adam
	from keras import regularizers

	model = Sequential()
	if wordvec is not None:
		wordvec_len = wordvec.shape[1]
		model.add(Embedding(vocab_size,
												wordvec_len,
												weights=[wordvec],
												trainable=False,
												mask_zero=True))
	else:
		model.add(Embedding(vocab_size,
												wordvec_len,
												mask_zero=True))

	model.add(Dropout(dropout))
	model.add(GRU(units=64, dropout=dropout, recurrent_dropout=dropout))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))

	adam = Adam()
	model.compile(loss='categorical_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])

	model.summary()
	return model


def predict_proba(model, data, all_lens):
	start, batch, probs = 0, 0, None

	num_data = len(data)
	while start < num_data:
		end = start + batch_size
		batch_x, lens = data[start:end], all_lens[start:end]
		batch_x = padd_zero(batch_x, max(lens))

		if probs is None:
			probs = model.predict_proba(batch_x, verbose=0)
		else:
			probs = np.vstack((probs, model.predict_proba(batch_x, verbose=0)))
		start += batch_size
		batch += 1

		if batch % 1000 == 0:
			print("Predicting proba for batch %d..." % batch)

	return probs


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_dir", help="Directory containing file 'train_data.txt'")
	parser.add_argument("kfold", help="K-fold cross validation", type=int)
	parser.add_argument("vector_size", help="Word embedding vector size", type=int)
	parser.add_argument("dropout", help="Dropout rate for RNN", type=float)
	parser.add_argument("--unlabeled", help="Unlabeled data for self-learning")
	parser.add_argument("--thresh", help="Confidence for accepting unlabeled data", type=float, default=0.9)
	parser.add_argument("--wordvec", help="Pretrained word vector file")

	return parser.parse_args()


def main():
	args = parse_input()

	train_dir = args.train_dir
	k_fold = args.kfold
	wordvec_len = args.vector_size
	dropout = args.dropout


	# perform k-fold cross validation
	if k_fold > 1:
		for fold in range(k_fold):
			cv_dir = os.path.join(train_dir, 'cv' + str(fold))
			if not os.path.isdir(cv_dir):
				print("Haven't performed fold_split. Exit!")
				exit()

			train_data_path = os.path.join(cv_dir, 'train_data.txt')
			train_label_path = os.path.join(cv_dir, 'train_label.txt')
			val_data_path = os.path.join(cv_dir, 'val_data.txt')
			val_label_path = os.path.join(cv_dir, 'val_label.txt')

			# Create model dir based on time now
			time_now = datetime.now().strftime('%m-%d_%H:%M')
			model_subdir = os.path.join(MODEL_DIR, time_now, str(k_fold) + "-" + str(fold))
			if not os.path.isdir(model_subdir):
				os.makedirs(model_subdir)

			# Create embedding file
			if args.wordvec is None:
				wordvec = None
				embedding_path = os.path.join(cv_dir, "embedding.txt")
				if not os.path.isfile(embedding_path):
					word_dict = gen_word_embeddings(train_data_path, cutoff=5)
					write_word_embeddings(word_dict, embedding_path)
				word_index = get_word_embeddings(embedding_path)
				shutil.copy2(embedding_path, model_subdir)
			# Or load pre-trained word vectors
			else:
				wordvec, word_index = load_wordvec(args.wordvec)
				shutil.copy2(args.wordvec, model_subdir)

			# Read labeled data and split into train and validation
			train_data, train_label, train_lens = read_data(train_data_path, train_label_path, word_index, handle_oov=True)
			val_data, val_label, val_lens = read_data(val_data_path, val_label_path, word_index, handle_oov=True)
			val_data = padd_zero(val_data, max(val_lens))

			# Load unlabeled data if applicable
			unlabeled_len = 0
			if args.unlabeled is not None:
				unlabeled, _, unlabeled_lens = read_data(args.unlabeled, None, word_index, handle_oov=True)
				'''
				unlabeled = unlabeled[:128]
				unlabeled_lens = unlabeled_lens[:128]
				'''

			'''
			filepath = os.path.join(model_subdir, 'Model.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
			checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=10, save_best_only=True, mode='auto')
			'''

			model = gru(len(word_index) + 1, wordvec_len=wordvec_len, dropout=dropout, wordvec=wordvec)

			# Train indefinitely until no unlabeled coule be confidently labeled
			epoch, best_acc = 0, 0.
			for _ in range(nb_epochs):
				t0 = time.time()
				start, total_loss, total_acc, batch = 0, 0., 0., 0

				num_data = len(train_data)
				while start < num_data:
					end = start + batch_size
					batch_x, batch_y, lens = train_data[start:end], train_label[start:end], train_lens[start:end]
					batch_x = padd_zero(batch_x, max(lens))

					loss, acc = model.train_on_batch(batch_x, batch_y)
					total_loss += loss * len(batch_x)
					total_acc += acc * len(batch_x)
					start += batch_size
					batch += 1
					if batch % 1000 == 0:
						print("Processed data %d... loss: %.4f, acc: %.4f" % (end, loss, acc))

				t1 = time.time()
				print("Epoch %d - elapsed time: %.2f, training loss: %.4f, acc: %.4f" % (epoch, (t1-t0), total_loss/num_data, total_acc/num_data))
				epoch += 1

				# Validation accuracy
				loss, acc = model.evaluate(val_data, val_label)
				print("\nValidation loss: %.4f, acc: %.4f" % (loss, acc))
				if acc > best_acc:
					filepath = os.path.join(model_subdir, 'Model.%02d-%.4f-%.4f.hdf5' % (epoch, acc, loss))
					print("Validation accuracy improved from %.4f to %.4f, saving model to %s..." % (best_acc, acc, filepath))
					model.save(filepath)
					best_acc = acc

				if args.unlabeled is not None:
					print(unlabeled.shape)
					print(unlabeled_lens.shape)
					pred = predict_proba(model, unlabeled, unlabeled_lens)
					print(pred.shape)

					confident_idx = np.any(pred > args.thresh, axis=1)
					print("%d unlabeled data pass threshold %.2f, adding to training data" % (np.sum(confident_idx), args.thresh))
					#print(confident_idx)
					#input("confident_idx")

					# Stack new data into training data
					train_data = np.concatenate((train_data, unlabeled[confident_idx]))

					# Stack new label into training labels
					new_label = np.around(pred[confident_idx])
					new_label = np.array(new_label)
					#print(new_label.shape)
					train_label = np.vstack((train_label, new_label))
					#print(train_label.shape)

					# Stack length information into training lengths
					train_lens = np.concatenate((train_lens, unlabeled_lens[confident_idx]))

					# Remove confident data
					confident_idx = np.where(confident_idx)
					unlabeled = np.delete(unlabeled, confident_idx, axis=0)
					unlabeled_lens = np.delete(unlabeled_lens, confident_idx, axis=0)

					# Shuffle training data
					shuf_index = np.random.permutation(len(train_data))
					train_data = train_data[shuf_index]
					train_label = train_label[shuf_index]
					train_lens = train_lens[shuf_index]
					#input("")


if __name__ == '__main__':
	main()
