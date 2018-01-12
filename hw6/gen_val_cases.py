from random import randint

num_labeled = 150
num_pairs = 1000

with open("data/val_cases.csv", "w") as outf:
	outf.write("ID,image1_index,image2_index\n")
	for idx in range(num_pairs):
		idx1 = randint(0, num_labeled-1)
		idx2 = idx1
		while idx2 == idx1:
			idx2 = randint(0, num_labeled-1)
		outf.write("%d,%d,%d\n" % (idx, randint(0, num_labeled-1), randint(0, num_labeled-1)))
