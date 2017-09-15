import sys

def main():
	argc = len(sys.argv)

	if argc != 3:
		print("Usage: python Q1_validate.py words_path produced_path")
		exit()

	word_path = sys.argv[1]
	q1_path = sys.argv[2]

	with open(word_path, "r") as inf:
		words = inf.readline().strip().split(" ")
		print("Total words in original file: %d" % len(words))
	
	total = 0
	with open(q1_path, "r") as inf:
		for line in inf:
			print(line + " abc")
			line = line.strip()
			word, idx, cnt = line.split(" ")
			total += int(cnt)

	print("Total words in produced file: %d" % total)
	print("Total words in original file: %d" % len(words))

if __name__ == '__main__':
	main()
