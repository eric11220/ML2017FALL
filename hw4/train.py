import os
import numpy as np
from datetime import datetime
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold

from data import *
from func import *

MODEL_DIR = 'models'

def main():
	np.set_printoptions(suppress=True)

	args = parse_input()
	x, y = load_data_label('data/train_data_no_punc.txt', 'data/train_label.txt')

	tokenizer = Tokenizer(num_words=args.top_words)
	tokenizer.fit_on_texts(x)

	if args.wordvec is not None:
		wordvec, veclen = load_wordvec(args.wordvec)

		import operator
		sorted_word_counts = sorted(tokenizer.word_counts.items(), key=operator.itemgetter(1), reverse=True)

		word_index = tokenizer.word_index
		embedding_matrix = np.zeros((args.top_words + 1, veclen))
		for word, count in sorted_word_counts[:args.top_words]:
			embedding_vector = wordvec.get(word, None)

			idx = word_index[word]
			if embedding_vector is not None:
				embedding_matrix[idx] = embedding_vector
			# words not found in embedding index will be treated as unknown. yet, should not enter here
			else:
				embedding_matrix[idx] = wordvec['<unk>']
	else:
		embedding_matrix = None

	if args.unlabeled is not None:
		X_unlabeled, _ = load_data_label(args.unlabeled, None, len_limit=40)
		X_unlabeled = tokenizer.texts_to_sequences(X_unlabeled)

	if args.modeldir is None:
		time_now = datetime.now().strftime('%m-%d_%H-%M')
		model_dir = os.path.join(MODEL_DIR, time_now)
	else:
		model_dir = args.modeldir

	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)

	skf = StratifiedKFold(n_splits=args.kfold)
	for cv_idx, (train_index, test_index) in enumerate(skf.split(x, y)):
		model_subdir = os.path.join(model_dir, "%s-%s" % (str(args.kfold), str(cv_idx)))
		if not os.path.isdir(model_subdir):
			os.makedirs(model_subdir)
		
		X_train = [x[idx] for idx in train_index]
		y_train = [y[idx] for idx in train_index]
		X_val = [x[idx] for idx in test_index]
		y_val = [y[idx] for idx in test_index]
		
		X_train = tokenizer.texts_to_sequences(X_train)
		X_val = tokenizer.texts_to_sequences(X_val)

		max_train_len = max([len(row) for row in X_train])
		max_val_len = max([len(row) for row in X_val])
		maxlen = max(max_train_len, max_val_len)
		if args.unlabeled is not None:
			max_unlabeled_len = max([len(row) for row in X_unlabeled])
			maxlen = max(maxlen, max_unlabeled_len)
			X_unlabeled = sequence.pad_sequences(X_unlabeled, maxlen=maxlen)
		

		X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding="pre")
		X_val = sequence.pad_sequences(X_val, maxlen=maxlen, padding="pre")
		
		model = create_model(args.top_words+1,
										args.embedding_vector_length,
										embedding_matrix,
										args.trainable,
										args.dropout)

		best_acc = 0.
		for epoch in range(10):
			hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=64)

			val_acc, val_loss = hist.history['val_acc'][0], hist.history['val_loss'][0]
			if val_acc > best_acc:
				path = os.path.join(model_subdir, "Model.epoch%d-%.4f-%.4f.hdf5" % (epoch, val_acc, val_loss))
				model.save(path)
				best_acc = val_acc

			if args.unlabeled is not None:
				new_X_train, new_y_train = get_new_data(model, X_unlabeled, args.thresh)
				X_train = np.vstack((X_train, new_X_train))
				y_train = np.concatenate((y_train, new_y_train))

		break


if __name__ == "__main__":
	main()
