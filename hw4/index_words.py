import sys

def gen_word_embeddings(*args, cutoff=10):
	word_dict = {}
	for path in args:
		with open(path, 'r') as inf:
			for line in inf:
				line = line.strip().split()
				for word in line:
					if word_dict.get(word, None) is None:
						word_dict[word] = 1
					else:
					 	word_dict[word] += 1

	to_delete_keys = []
	for key, value in sorted(word_dict.items(), key=lambda x:x[1]):
		if value < cutoff:
			to_delete_keys.append(key)

	for key in to_delete_keys:
		del(word_dict[key])

	return word_dict


def write_word_embeddings(word_dict, path):
	with open(path, 'w') as outf:
		for key, value in sorted(word_dict.items(), key=lambda x:x[1]):
			outf.write("%s %d\n" % (key, value))


def get_word_embeddings(path):
	word_dict, cnt = {}, 1
	with open(path, 'r') as inf:
		for line in inf:
			word, _ = line.strip().split(' ')
			word_dict[word] = cnt
			cnt += 1

	return word_dict


def main():
	'''
	all_data = []
	for path in sys.argv[1:]:
		with open(path, 'r') as inf:
			all_data.append(inf.readlines())
	'''

	word_dict = gen_word_embedding(sys.argv[1:])
	write_word_embeddings(word_dict, 'embedding_index_%d.txt' % cutoff)


if __name__ == '__main__':
	main()
