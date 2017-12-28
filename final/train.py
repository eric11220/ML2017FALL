import numpy as np
from loader import *
from wordvec import *
from model import *
wordvec_path = "wordvecs/wordvec_dim150_mincount3"

batch_size = 128

def main():
	first_sents, second_sents, labels, word_index = get_train_sents()
	wordvec = load_wordvec(path=wordvec_path)
	embedding = wordvec_to_embedding(word_index, wordvec)

	perm = np.random.permutation(len(first_sents))
	first_sents = first_sents[perm]
	second_sents = second_sents[perm]
	labels = labels[perm]

	val_start = int(len(first_sents) * 0.9)
	train_x1 = first_sents[:val_start, :]
	train_x2 = second_sents[:val_start, :]
	train_y = labels[:val_start]

	val_x1 = first_sents[val_start:, :]
	val_x2 = second_sents[val_start:, :]
	val_y = labels[val_start:]

	model = build_model(first_sents.shape[1], embedding)
	model.fit(
		[train_x1, train_x2],
		train_y,
		validation_data=([val_x1, val_x2], val_y),
		batch_size=batch_size)


if __name__ == '__main__':
	main()
