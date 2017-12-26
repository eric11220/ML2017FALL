import os
import jieba
import numpy as np
from gensim.models import word2vec as w2v
from keras.utils.np_utils import to_categorical
wordvec_dir = "wordvecs"

# 0 for <PAD>
def onehot_embedding(tokenizer):
	word_index = tokenizer.word_index

	# last one for <SOS>
	onehot = np.arange(len(word_index) + 2)
	onehot = to_categorical(onehot)
	return onehot


def wordvec_to_embedding(tokenizer, wordvec):
	word_index = tokenizer.word_index
	num_words = len(word_index) + 1

	embeddings = None
	for word, idx in word_index.items():
		vec = wordvec[word]
		if embeddings is None:
			embeddings = np.zeros((num_words, vec.shape[0]))
		embeddings[idx] = vec
	return embeddings


def load_wordvec(path=os.path.join(wordvec_dir, "wordvec_100")):
	return w2v.Word2Vec.load(path)


def train_wordvec(sentences, dim=100):
	print("Training word vectors")
	word_vec = w2v.Word2Vec(sentences, size=dim, min_count=0)
	word_vec.save(os.path.join(wordvec_dir, "wordvec_%d" % dim))


def get_all_sentences(min_count=5):
	jieba.set_dictionary('jieba/dict.txt.big')

	words, count = [], {}
	with open("data/training_data/all_train.txt", "r") as inf:
		for line in inf:
			line = line.strip()
			line = "".join([c for c in line if u'\u4e00' <= c <= u'\u9fff'])
			line = jieba.cut(line, cut_all=False)

			sent = []
			for c in line:
				count[c] = 1 if c not in count else count[c] + 1
				sent.append(c)
			words.append(sent)

	# Replace words appearing less than min_count as unk
	sentences = []
	for sent in words:
		converted = []
		for word in sent:
			word = "unk" if count[word] < min_count else word
			converted.append(word)
		sentences.append(converted)
	return sentences


def main():
	sentences = get_all_sentences()
	word_vec = train_wordvec(sentences)


if __name__ == '__main__':
	main()
