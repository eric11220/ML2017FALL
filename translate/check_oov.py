import jieba
from parse_wordvec import load_wordvec

wordvec = load_wordvec("wordvec/simplified.pickle", pickle=True)
print("Wordvec loaded")

word_cnt, word_oov, char_cnt, char_oov = 0, 0, 0, 0
with open("data/train.caption", "r") as inf:
	for line in inf:
		line = line.strip()
		chars = line.split(" ")
		for ch in chars:
			char_cnt += 1
			if ch not in wordvec:
				char_oov += 1

		line = line.replace(" ", "")
		seg_list = jieba.cut(line, cut_all=False)
		for word in seg_list:
			word_cnt += 1
			if word not in wordvec:
				word_oov += 1

print("Word level oov: %.4f, char level oov: %.4f"
				% (word_oov/word_cnt, char_oov/char_cnt))
