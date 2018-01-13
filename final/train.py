import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint
from loader import *
from wordvec import *
from model import *
modeldir = "models"


def parse_inputs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wordvec", help="Pre-trained word vector path", default="wordvecs/wordvec_dim150_mincount3")
	parser.add_argument("--batch_size", help="Batch size", default=128)
	parser.add_argument("--n_epoch", help="Number of epoch", default=10)
	parser.add_argument("--num_sent", help="Number of sentences", default=2)
	return parser.parse_args()


def main():
	args = parse_inputs()

	first_sents, second_sents, labels, word_index = get_train_sents(num_sent=args.num_sent)
	print(first_sents[:10])
	input("")
	wordvec = load_wordvec(path=args.wordvec)
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

	wordvec_name = os.path.basename(args.wordvec)
	subdir = os.path.join(modeldir, "%s_numsent%s" % (wordvec_name, args.num_sent))
	os.makedirs(subdir, exist_ok=True)

	filepath = os.path.join(subdir, 'Model.{epoch:02d}-{acc:.4f}-{val_acc:.4f}-{val_loss:.4f}.hdf5')
	ckpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

	model = build_model(first_sents.shape[1], embedding)
	model.fit(
		[train_x1, train_x2],
		train_y,
		validation_data=([val_x1, val_x2], val_y),
		epochs=args.n_epoch,
		batch_size=args.batch_size,
		callbacks=[ckpointer])


if __name__ == '__main__':
	main()
