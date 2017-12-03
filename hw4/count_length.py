lens = {}
with open('data/training_nolabel.txt', 'r') as inf:
	for idx, line in enumerate(inf):
		line = line.strip().split(' ')
		length = len(line)
		if lens.get(length, None) is None:
			lens[length] = 1
		else:
			lens[length] += 1

cnt = 0
for key, _ in lens.items():
	cnt += lens[key]
	lens[key] = cnt / idx

for key, val in lens.items():
	print(key, val)
