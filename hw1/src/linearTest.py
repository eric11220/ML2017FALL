import sys
import os
import numpy as np

coef_file = '../coefficients/num_prev_hour9_1_0'

def predict(coef, data):
	return np.sum(coef * data)

def append_deg(coef, data):
	coef_len = len(coef) - 1
	data_len = len(data)

	deg = 2
	while len(data) < coef_len:
		data = np.append(data, data ** deg)
		deg += 1

	return data

def predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, coef):
	with open(out_csv, 'w') as outf:
		outf.write("id,value\n")
		with open(in_csv, 'r') as inf:
			cur_id, all_feats = None, {}
			for line in inf:
				vals = line.strip().split(',')
				id, feat_name, feat = vals[0], vals[1], vals[2:]

				for idx, f in enumerate(feat):
					try:
						_ = float(f)
					except:
						feat[idx] = 0

				if cur_id == None or cur_id == id:
					all_feats[feat_name] = feat
					cur_id = id
				else:
					final_feat = []
					for name in feat_order:
						final_feat.extend(all_feats[name])

					final_feat = np.asarray(final_feat, dtype=np.float32)
					final_feat = append_deg(coef, final_feat)

					final_feat = (final_feat - train_mean) / train_std
					final_feat = np.append(final_feat, 1.0)

					prediction = predict(coef, final_feat)
					outf.write(cur_id + "," + str(prediction) + "\n")
					cur_id, all_feats = id, {feat_name: feat}

		final_feat = []
		for name in feat_order:
			final_feat.extend(all_feats[name])
		final_feat = np.asarray(final_feat, dtype=np.float32)
		final_feat = append_deg(coef, final_feat)

		final_feat = (final_feat - train_mean) / train_std
		final_feat = np.append(final_feat, 1.0)

		prediction = predict(coef, final_feat)
		outf.write(cur_id + "," + str(prediction) + "\n")

def load_parameters():
	with open(coef_file, 'r') as inf:
		lines = inf.readlines()

		feat_order = lines[0].strip().split(' ')
		train_mean = np.asarray(lines[1].strip().split(' '), dtype=np.float32)
		train_std = np.asarray(lines[2].strip().split(' '), dtype=np.float32)
		coef = np.asarray(lines[3].strip().split(' '), dtype=np.float32)

	return feat_order, train_mean, train_std, coef

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python linearReg.py input_csv output_csv")
		exit()

	in_csv = sys.argv[1]
	out_csv = sys.argv[2]

	feat_order, train_mean, train_std, coef = load_parameters()
	test_data = predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, coef)

if __name__ == '__main__':
	main()
