import sys

def main():
	argc = len(sys.argv)
	if argc != 2:
		print("Usage: python q1.py path")
		exit()
	
	word_path = sys.argv[1]	

	word_list, cnt_dict = [], {}
	with open(word_path, "r") as inf:
		line = inf.readline().strip()
		words = line.split(" ")
		
		for word in words:
			if word not in word_list:
				word_list.append(word)
			if cnt_dict.get(word, None) is None:
				cnt_dict[word] = 1
			else:
				cnt_dict[word] = cnt_dict[word] + 1
	
	with open("Q1.txt", "w") as outf:
		for idx, word in enumerate(word_list):
			outf.write("%s %d %d" % (word, idx, cnt_dict[word]))
			if idx != len(word_list) - 1:
				outf.write("\n")

if __name__ == '__main__':
	main()
