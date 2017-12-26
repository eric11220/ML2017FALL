import random
import argparse
import _pickle as pickle
from datetime import datetime
from keras.utils.np_utils import to_categorical

from loader import *
from model import *
from wordvec import *
modeldir = 'models'


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wordvec", help="Pre-trained word vector path", default=None)
	parser.add_argument("--batch_size", help="Batch size", default=64)
	parser.add_argument("--n_epoch", help="Number of epoch", default=30)
	parser.add_argument("--num_sent", help="Number of sentences to care", default=2)
	return parser.parse_args()


def run(model, x, y, num_words, batch_size, num_sent=1, validation=False, display_freq=20, shuffle=True):

	indices = []
	start, end = 0, batch_size
	while end + num_sent < len(x):
		indices.append(start)
		forward = batch_size - num_sent + 1
		start += forward; end += forward

	if shuffle is True:
		random.shuffle(indices)

	num_batch, losses = 0, 0
	for start in indices:
		model.reset_states()
		end = start + batch_size
		for time in range(num_sent):
			batch_encode = x[start+time:end+time]
			batch_decode = y[start+time+1:end+time+1]

			batch_target = y[start+time+1:end+time+1, 1:]
			batch_target = to_categorical(batch_target, num_classes=num_words)
			batch_target = np.reshape(batch_target, (batch_size, -1, num_words))

			if validation is True:
				loss = model.evaluate(x=[batch_encode, batch_decode], y=batch_target, batch_size=batch_size)
			else:
				loss = model.train_on_batch([batch_encode, batch_decode], batch_target)
		losses += loss

		num_batch += 1 
		if num_batch % display_freq == 0 and validation is False:
			print("Processed %d sentences, loss: %.4f" % (num_batch * batch_size, losses / num_batch))

	epoch_loss = losses / num_batch
	return epoch_loss


def main():
	args = parse_input()
	batch_size = args.batch_size

	# Get sentences
	sentences, tokenizer, maxlen = get_training_sentences(wordvec_path=args.wordvec)

	# Create model
	onehot = onehot_embedding(tokenizer)
	if args.wordvec is not None:
		wordvec = load_wordvec()
		embeddings = wordvec_to_embedding(tokenizer, wordvec)
	else:
		embeddings = onehot
	num_words = embeddings.shape[0]
	model = seq2seq(sentences.shape[1], num_words, embeddings, onehot, batch=batch_size)
	print("Model constructed")

	# Ready data
	x, y = sentences, np.array(sentences)
	sos = np.asarray([[num_words-1] * len(sentences)]).T
	y = np.concatenate((sos, y), axis=1)

	# last 5% as validation data
	num_train = int(len(sentences) * 0.99)
	train_x, train_y = x[:num_train], y[:num_train]
	val_x, val_y = x[num_train:], y[num_train:]

	# Training
	epoch, best_loss = 0, 100
	time_now = datetime.now().strftime('%m-%d-%H-%M')
	model_subdir = os.path.join(modeldir, time_now)
	os.makedirs(model_subdir, exist_ok=True)

	with open(os.path.join(model_subdir, "tokenizer.pickle"), 'wb') as handle:
		pickle.dump([tokenizer, maxlen], handle)

	while epoch <= args.n_epoch:
		epoch_loss = run(model, train_x, train_y, num_words, batch_size, args.num_sent)
		val_loss = run(model, val_x, val_y, num_words, batch_size, validation=True)
		if val_loss < best_loss:
			best_loss = val_loss
			model.save(os.path.join(model_subdir, "Epoch%d_loss%.4f_valloss_%.4f.hdf5" % (epoch, epoch_loss, val_loss)))
		print("Epoch %d loss %.4f, val_loss: %.4f" % (epoch, epoch_loss, val_loss))
		epoch += 1


if __name__ == '__main__':
	main()
