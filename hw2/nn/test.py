import sys
import os
import numpy as np

from keras.models import model_from_json

def predict(model, data):
	return model.predict_classes(data)


def predict_csv(in_csv, out_csv, model_path):
	weight_path = model_path + '_weight.h5'
	mean_std_path = model_path + '_mean_std.npy'

	# Load training mean and std
	train_mean, train_std = np.load(mean_std_path, encoding='latin1')

	# Load NN model_path
	json_file = open(model_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(weight_path)

	with open(out_csv, 'w') as outf:
		outf.write("id,label\n")
		with open(in_csv, 'r') as inf:
			inf.readline()
			cur_id = 1
			for line in inf:
				feats = line.strip().split(',')
				feats = np.asarray(feats, dtype=np.float32)
				feats = (feats - train_mean) / train_std
				feats = np.reshape(feats, [1, -1])

				# NN predict
				prediction = predict(model, feats)

				outf.write(str(cur_id) + "," + str(prediction[0]) + "\n")
				cur_id += 1


def main():
	argc = len(sys.argv)
	if argc != 4:
		print("Usage: python test.py input_csv output_csv model_path")
		exit()

	in_csv = sys.argv[1]
	out_csv = sys.argv[2]
	model_path = sys.argv[3]

	test_data = predict_csv(in_csv, out_csv, model_path)

if __name__ == '__main__':
	main()
