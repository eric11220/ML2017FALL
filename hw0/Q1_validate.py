import sys

def main():
	argc = len(sys.argv)

	if argc != 3:
		print("Usage: python Q1_validate.py words_path produced_path")
		exit()

	word_path = sys.argv[1]
	q1_path = sys.argv[2]

	unique_words = []
	with open(word_path, "r") as inf:
		words = inf.readline().strip().split(" ")
		for word in words:
			if word not in unique_words:
				unique_words.append(word)
	
	total, word_cnt = 0, 0
	with open(q1_path, "r") as inf:
		for line in inf:
			print(line + "abc")
			word_cnt += 1
			line = line.strip()
			word, idx, cnt = line.split(" ")
			total += int(cnt)

	print("unique words in original file %d" % len(unique_words))
	print("words in produced file %d" % word_cnt)

	print("Total words in produced file: %d" % total)
	print("Total words in original file: %d" % len(words))

if __name__ == '__main__':
	main()
