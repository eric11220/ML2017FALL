import sys
import os
import numpy as np

from splitData import read_data

epsilon = 1e-8

def predict(model, data, model_type):
	if model_type == "nn":
		return model.predict_classes(data)
	elif model_type == "svc":
		return model.predict(data)


def predict_csv(in_csv, out_csv, model_path, model_type):
	mean_std_path = model_path + '_mean_std.npy'

	# Load training mean and std
	mean, std = np.load(mean_std_path, encoding='latin1')

	# Load NN model_path
	if model_type == "nn":
		from keras.models import model_from_json
		weight_path = model_path + '_weight.h5'
		json_file = open(model_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights(weight_path)
	elif model_type == "svc":
		from sklearn.externals import joblib
		model = joblib.load(model_path)

	numeric_indices, data = read_data(in_csv)
	numeric_indices = np.asarray(numeric_indices, dtype=np.uint8)

	train_mean = np.zeros((mean.shape[0], ))
	train_std = np.ones((std.shape[0], ))
	train_mean[numeric_indices] = mean[numeric_indices]
	train_std[numeric_indices] = std[numeric_indices]

	with open(out_csv, 'w') as outf:
		outf.write("id,label\n")
		cur_id = 1
		for feats in data:
			feats = np.asarray(feats, dtype=np.float32)
			feats = (feats - train_mean) / (train_std + epsilon)
			feats = np.reshape(feats, [1, -1])

			prediction = predict(model, feats, model_type)

			outf.write(str(cur_id) + "," + str(prediction[0]) + "\n")
			cur_id += 1


def main():
	argc = len(sys.argv)
	if argc != 5:
		print("Usage: python test.py input_csv output_csv model_path model_type")
		exit()

	in_csv = sys.argv[1]
	out_csv = sys.argv[2]
	model_path = sys.argv[3]
	model_type = sys.argv[4]

	test_data = predict_csv(in_csv, out_csv, model_path, model_type)

if __name__ == '__main__':
	main()
