import numpy as np
import _pickle as pk


wordvec_path = "wordvec/wiki.zh.vec"
simplified_wordvec_path = "wordvec/simplified.vec"
word_path = "wordvec/words"
simplified_word_path = "wordvec/simplified_words"
simplified_pickle = "wordvec/simplified.pickle"


def get_all_chinese_words():
	words = []
	with open(wordvec_path, "r") as inf:
		for line in inf:
			chinese = line.strip().split(" ", 1)[0]
			words.append(chinese)
	
	with open(word_path, "w") as outf:
		for word in words:
			outf.write(word + "\n")


def simplify_chinese_words():
	with open(simplified_wordvec_path, "w") as outf:
		with open(wordvec_path, "r") as inf:
			header = inf.readline()
			outf.write(header)
			with open(simplified_word_path, "r") as inf2:
				for line1, line2 in zip(inf, inf2):
					simplified_word = line2.strip()
					vec = line1.strip().split(" ", 1)[1]
					outf.write(simplified_word + " " + vec + "\n")


def load_wordvec(path, pickle=False):
	if pickle is False:
		wordvec = {}
		with open(path, "r") as inf:
			cnt = 0
			for line in inf:
				word, vec_line = line.strip().split(" ", 1)
				if all([u'\u4e00' <= c <= u'\u9fff' for c in word]):
					print(word)
					cnt += 1
					vec = vec_line.split(" ")
					wordvec[word] = np.asarray(vec, dtype=np.float32)
	else:
		with open(path, "rb") as inf:
			wordvec = pk.load(inf)
	return wordvec


def add_oov_to_wordvec(wordvec):
	avg, cnt = None, 0
	for vec in wordvec.values():
		cnt += 1
		if avg is None:
			avg = vec
		else:
			avg += vec

	wordvec["oov"] = avg/cnt


def main():
	#simplify_chinese_words()
	wordvec = load_wordvec(simplified_wordvec_path, pickle=False)
	with open(simplified_pickle, 'wb') as outf:
		pk.dump(wordvec, outf)
	wordvec = load_wordvec(simplified_pickle, pickle=True)
	input("loaded")
	print(wordvec["研究"])


if __name__ == '__main__':
	main()
