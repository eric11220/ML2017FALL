# python3 train.py data/processed/ 5

import sys
import keras
import shutil
from datetime import datetime

from splitData import *
from data.index_words import *

nb_classes = 2
batch_size = 32
nb_epochs = 20
MODEL_DIR = 'models'

def gru(vocab_size, max_caption_len, word_vector_len=256):
	from keras.models import Sequential
	from keras.layers import Embedding, GRU, Dense, Flatten
	from keras.optimizers import Adam

	model = Sequential()
	model.add(Embedding(vocab_size, word_vector_len, input_length=max_caption_len, mask_zero=True))
	model.add(GRU(units=int(word_vector_len * 0.5)))
	model.add(Dense(nb_classes, activation='softmax'))

	adam = Adam()
	model.compile(loss='categorical_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])

	model.summary()
	return model


def main():
	argc = len(sys.argv)
	if argc != 4 and argc != 5:
		print("usage: python train.py training_dir fold vector_size (unlabel_data_path)")
		exit()

	train_dir = sys.argv[1]
	k_fold = int(sys.argv[2])
	word_vector_len = int(sys.argv[3])
	if argc == 5:
		unlabel_train_path = sys.argv[3]

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

			# Create embedding file
			embedding_path = os.path.join(cv_dir, "embedding.txt")
			if not os.path.isfile(embedding_path):
				word_dict = gen_word_embeddings(train_data_path)
				write_word_embeddings(word_dict, embedding_path)

			word_index = get_word_embeddings(embedding_path)
			train_data, train_label, train_len = read_data(train_data_path, train_label_path, word_index, handle_oov=True)
			val_data, val_label, val_len = read_data(val_data_path, val_label_path, word_index, handle_oov=True)

			# Padding train and validation data to max_len
			max_len = train_len if train_len > val_len else val_len
			train_data = padd_zero(train_data, max_len)
			val_data = padd_zero(val_data, max_len)

			time_now = datetime.now().strftime('%m-%d_%H:%M')
			model_subdir = os.path.join(MODEL_DIR, time_now, str(k_fold) + "-" + str(fold))
			if not os.path.isdir(model_subdir):
				os.makedirs(model_subdir)
			shutil.copy2(embedding_path, model_subdir)

			filepath = os.path.join(model_subdir, 'Model.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
			checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=10, save_best_only=True, mode='auto')

			model = gru(len(word_index) + 2, max_len, word_vector_len=word_vector_len)
			model.fit(
				train_data, 
				train_label,
				batch_size=batch_size,
				epochs=nb_epochs,
				validation_data=(val_data, val_label),
				callbacks=[checkpointer]
			)


if __name__ == '__main__':
	main()
