import jieba
import numpy as np
from parse_wordvec import load_wordvec, add_oov_to_wordvec


def pad_sequences(word_seq, maxlen):
	num_seq, veclen = len(word_seq), len(word_seq[0][0])
	pad_seq = np.zeros((num_seq, maxlen, veclen))
	for idx, seq in enumerate(word_seq):
		content_len = len(seq)
		pad_seq[idx, -content_len:, :] = np.asarray(seq)
	return pad_seq


def vectorize_caption(path="data/train.caption", wordvec_pickle="wordvec/simplified.vec"):
	wordvec = load_wordvec("wordvec/simplified.pickle", pickle=True)
	add_oov_to_wordvec(wordvec)

	word_seq, maxlen = [], 0
	with open(path, "r") as inf:
		for line in inf:
			line = line.strip().replace(" ", "")
			seg_list = jieba.cut(line, cut_all=False)

			tmp_seq = []
			for word in seg_list:
				vec = wordvec.get(word, None)
				if vec is None:
					vec = wordvec["oov"]
				tmp_seq.append(vec)

			if len(tmp_seq) > maxlen:
				maxlen = len(tmp_seq)
			word_seq.append(tmp_seq)

	word_seq = pad_sequences(word_seq, maxlen)
	return word_seq


def main():
	word_seq = vectorize_caption()


if __name__ == '__main__':
	main()
