import os
import sys
import argparse
import numpy as np
import _pickle as pickle
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from loader import *
np.set_printoptions(suppress=True)


def compute_loss(predict, goal, num_words):
	losses = []
	for sent_idx, (p, g) in enumerate(zip(predict, goal)):
		loss, sent_len = 0., 0
		for p_word, idx in zip(p, g):
			idx = int(idx)
			val = p_word[idx]
			if idx != 0:
				loss += -np.log(val + 1e-7)
				sent_len += 1

		if sent_len == 0:
			loss = 1e7
		else:
			loss /= sent_len
		losses.append(loss)
	return losses


def write_answers_to_file(answers, out_path="results/test.csv"):
	with open(out_path, 'w') as outf:
		outf.write("id,ans\n")
		for idx, ans in enumerate(answers):
			if idx > 5060:
				break
			outf.write("%d,%d\n" % (idx+1, ans))


def testing(model, questions, options, all_targets, maxlen, num_words, num_sent=1, batch_size=64):
	model.summary()

	encode = np.zeros((batch_size, maxlen))
	decode = np.zeros((batch_size, maxlen+1))
	goal = np.zeros((batch_size, maxlen))

	answers, start = None, 0
	while start <= len(questions):
		end = start + batch_size
		if end > len(questions):
			end = len(questions)
			encode = np.zeros((batch_size, maxlen))
			decode = np.zeros((batch_size, maxlen+1))
			goal = np.zeros((batch_size, maxlen))

		losses = []
		for choice in range(len(options[0])):
			model.reset_states()

			for idx in range(end - start):
				decode[idx] = options[start+idx][choice][0] 
				goal[idx] = all_targets[start+idx][choice][0]
				for sent_idx in range(1, num_sent+1):
					encode[idx] = questions[start+idx][-sent_idx]

			predicted = model.predict([encode, decode], batch_size=batch_size)
			print(start, end)
			loss = compute_loss(predicted, goal, num_words)
			losses.append(loss)
		losses = np.asarray(losses)
		ans = np.argmin(losses, axis=0)
		if answers is None:
			answers = ans
		else:
			answers = np.concatenate((answers, ans))
		start += batch_size
	return answers


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", help="Pre-trained model path", default="models/12-25-01-33/Epoch4_loss1.4615.hdf5")
	parser.add_argument("--test_path", help="Testing data path", default="data/testing_data.csv")
	return parser.parse_args()


def main():
	args = parse_input()
	model_path = args.model_path
	test_path = args.test_path
	
	model = load_model(model_path)
	model_dir = os.path.dirname(model_path)
	with open(os.path.join(model_dir, "tokenizer.pickle"), 'rb') as handle:
		tokenizer, maxlen = pickle.load(handle)

	num_words = len(tokenizer.word_index) + 2
	questions, options, targets = get_testing_sentences(tokenizer, maxlen)
	answers = testing(model, questions, options, targets, maxlen, num_words)
	write_answers_to_file(answers)


if __name__ == '__main__':
	main()
