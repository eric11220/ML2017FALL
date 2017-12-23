import os
import jieba
import numpy as np
import _pickle as pickle
from gensim.models import word2vec as w2v
from keras.preprocessing.text import Tokenizer


def count_sentence_lengths():
	qlen_dist, olen_dist = {}, {}
	with open("data/testing_data.csv", "r") as inf:
		inf.readline()
		for line in inf:
			_, q, options = line.strip().split(',')
			q = q.replace("A:", "").replace("B:", "").replace("\t", " ").split(' ')
			qlen = len(q)
			if qlen_dist.get(qlen, None) is None:
				qlen_dist[qlen] = 1
			else:
				qlen_dist[qlen] += 1
	
			options = options.replace("A:", "").replace("B:", "")
			options = options.split("\t")
			options = [o.split(' ') for o in options]
			for o in options:
				olen = len(o)
				if olen_dist.get(olen, None) is None:
					olen_dist[olen] = 1
				else:
					olen_dist[olen] += 1
	print(qlen_dist, olen_dist)


def padd_to_maxlen(vectors, maxlen, veclen):
	padded = np.zeros((len(vectors), maxlen, veclen))
	for idx, vec in enumerate(vectors):
		padded[idx, :len(vec), :] = vec
	return padded


def get_training_sentences(wordvec_path="wordvecs/wordvec_100", 
								presave_dir="data/training_vectors", training_dir = "data/training_data"):

	wordvec = w2v.Word2Vec.load(wordvec_path)

	all_vectors = []
	file_idx, maxlen = 0, 0
	for f in sorted(os.listdir(training_dir)):
		if f == "all_train.txt" or ".txt" not in f:
			continue

		name, ext = os.path.splitext(f)
		presave_path = os.path.join(presave_dir, "%s.pickle" % name)
		if os.path.isfile(presave_path):
			print("Loading from presaved pickle %s" % presave_path)
			with open(presave_path, 'rb') as handle:
				vectors = pickle.load(handle)
			all_vectors.append(vectors)
		else:
			vectors = []
			path = os.path.join(training_dir, f)
			print("Loading sentences form %s" % path)
			with open(path, 'r') as inf:
				for idx, line in enumerate(inf):

					# Remove non-chinese characters and transform to vectors
					tmp_vectors = []
					line = "".join([c for c in line.strip() if u'\u4e00' <= c <= u'\u9fff'])
					line = jieba.cut(line, cut_all=False)
					for word in line:
						vec = wordvec['unk'] if word not in wordvec else wordvec[word]
						tmp_vectors.append(vec)

					# Keep track of maxlen
					line_len = len(tmp_vectors)
					if line_len > maxlen:
						maxlen = line_len
					if line_len > 0:
						vectors.append(tmp_vectors)

			# Presave transformed vectors for later easy-loading
			print("Pickle dumping to %s" % presave_path)
			with open(presave_path, 'wb') as handle:
				pickle.dump(vectors, handle)

			all_vectors.append(vectors)
			file_idx += 1

	# When loading
	if maxlen == 0:
		maxlen = max([len(v) for vector in all_vectors for v in vector])

	# Padd to even length
	for idx, f_vectors in enumerate(all_vectors):
		all_vectors[idx] = padd_to_maxlen(f_vectors, maxlen, len(vectors[0][0]))
	return all_vectors
