import sys
import numpy as np
from collections import Counter

def read_labels(in_csv):
	with open(in_csv, "r") as inf:
		header = inf.readline()
		labels = [line.strip().split(',')[1] for line in inf]

	return header, labels


def ensemble(labels):
	labels = np.asarray(labels, dtype=np.str)
	labels = labels.T

	counts = [Counter(l).most_common(1)[0][0] for l in labels]
	return counts


def main():
	out_csv = sys.argv[-1]

	with open(out_csv, "w") as outf:
		labels = []
		for name in sys.argv[1:-1]:
			if ".csv" not in name:
				model, _ = os.path.splitext(os.path.basename(name))
				csv = model + ".csv"

				cmd = "python3 test.py %s data/test.csv results/%s.csv 0 1" % (name, model)
				os.system(cmd)
			else:
				csv = name
				print(csv)
			header, lbl = read_labels(csv)
			labels.append(lbl)

		labels = ensemble(labels)

		outf.write(header)
		for idx, label in enumerate(labels):
			outf.write(str(idx) + ',' + str(label) + "\n")


if __name__ == '__main__':
	main()
