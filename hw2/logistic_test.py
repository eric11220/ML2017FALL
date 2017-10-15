import sys
import os
import numpy as np

def predict(coef, train_data):
	logits = 1 / (1 + np.exp(-np.sum(coef * train_data))) > 0.5
	logits = logits.astype(int)
	return logits


def predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, coef):
	with open(out_csv, 'w') as outf:
		outf.write("id,label\n")
		with open(in_csv, 'r') as inf:
			inf.readline()
			cur_id = 1
			for line in inf:
				feats = line.strip().split(',')
				feats = np.asarray(feats, dtype=np.float32)
				feats = (feats - train_mean) / train_std
				feats = np.append(feats, 1.0)

				prediction = predict(coef, feats)

				outf.write(str(cur_id) + "," + str(prediction) + "\n")
				cur_id += 1


def load_parameters(coef_file):
	feat_order, train_mean, train_std, coef = np.load(coef_file, encoding='latin1')
	return feat_order, train_mean, train_std, coef


def main():
	argc = len(sys.argv)
	if argc != 4:
		print("Usage: python test.py input_csv output_csv coef_file")
		exit()

	in_csv = sys.argv[1]
	out_csv = sys.argv[2]
	coef_file = sys.argv[3]

	feat_order, train_mean, train_std, coef = load_parameters(coef_file)
	test_data = predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, coef)

if __name__ == '__main__':
	main()
