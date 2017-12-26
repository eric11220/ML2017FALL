import os
import jieba
import numpy as np
import _pickle as pickle
from gensim.models import word2vec as w2v
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def sentence_clean(sent, tokenizer):
	cleaned = ""
	for word in sent:
		if u'\u4e00' <= word <= u'\u9fff':
			if word not in tokenizer.word_index:
				cleaned += "unk "
			else:
				cleaned += "%s " % word
	return cleaned


# Longest test: 34 Chinese words
def get_testing_sentences(tokenizer, maxlen, path="data/testing_data.csv"):
	num_words = len(tokenizer.word_index) + 2
	questions, options, targets = [], [], []
	with open(path, "r") as inf:
		header = inf.readline()
		for line in inf:
			_, tmp_q, tmp_options = line.strip().split(',')
			tmp_q = tmp_q.replace("A:", "").replace("B:", "").replace("\t", " ").split(' ')
			q = tokenizer.texts_to_sequences([sentence_clean(sent, tokenizer) for sent in tmp_q])
			q = pad_sequences(q, maxlen=maxlen, padding="post")
			questions.append(q)
	
			tmp_options = tmp_options.replace("A:", "").replace("B:", "").split('\t')
			option, target = [], []
			for tmp_o in tmp_options:
				o = tokenizer.texts_to_sequences([sentence_clean(sent, tokenizer) for sent in tmp_o.split(" ")])
				sos = np.asarray([[num_words-1] * len(o)]).T 
				o = pad_sequences(o, maxlen=maxlen, padding="post")
				target.append(o)
				option.append(np.concatenate((sos, o), axis=1))
			targets.append(target)
			options.append(option)

		return questions, options, targets


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


def get_training_sentences(presave_path="data/training_vectors.pickle", wordvec_path="wordvecs/wordvec_100", min_count=5):
	if os.path.isfile(presave_path):
		print("Pickle loading")
		with open(presave_path, 'rb') as handle:
			return pickle.load(handle)

	else:
		# Word level training
		if wordvec_path is not None:
			jieba.set_dictionary("jieba/dict.txt.big")
			wordvec = w2v.Word2Vec.load(wordvec_path)

		# Character level
		else:
			wordvec = None

		sentences, count = [], {}
		with open("data/training_data/all_train.txt", "r") as inf:
			for line in inf:
				if wordvec is not None:
					line = jieba.cut(line.strip(), cut_all=False)
					line = " ".join(['unk' if word not in wordvec else word for word in line])
				else:
					sentence = ""
					for word in line:
						if u'\u4e00' <= word <= u'\u9fff':
							count[word] = count[word] + 1 if word in count else 1
							sentence += word + " "
					sentence = sentence.strip()
				sentences.append(sentence)

		for idx in range(len(sentences)):
			sent = sentences[idx]
			sifted = ""
			for word in sent:
				sifted += word if word == ' ' else ('unk' if count[word] < min_count else word)
			sentences[idx] = sifted

		'''
		for cnt in range(10):
			print(len([ch for ch, c in count.items() if c > cnt]))
		input("")
		'''

		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(sentences)

		sentences = tokenizer.texts_to_sequences(sentences)
		maxlen = max([len(s) for s in sentences])
		sentences = pad_sequences(sentences, maxlen=maxlen, padding="post")

		if not os.path.isfile(presave_path):
			print("Pickle dumping for later easy-loading")
			with open(presave_path, 'wb') as handle:
				pickle.dump([sentences, tokenizer, maxlen], handle)
		
	return sentences, tokenizer, maxlen
