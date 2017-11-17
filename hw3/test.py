from __future__ import print_function

import sys

import dataprocessing
from splitData import *

img_rows, img_cols, nb_classes = 48, 48, 7


def ConvertTo3DVolume(data):
	data = np.asarray(data) 
	data = np.reshape(data, [-1, img_rows, img_cols])
	data = data[:, :, :, np.newaxis]
	data = data.astype('float32')

	return data


def predict_prob(number, data, model, dim):
	toreturn = []
	for data5 in data:
		if number == 0:
			toreturn.append(dataprocessing.Flip(data5, dim))
		elif number == 1:
			toreturn.append(dataprocessing.Roated15Left(data5, dim))
		elif number == 2:
			toreturn.append(dataprocessing.Roated15Right(data5, dim))
		elif number == 3:
			toreturn.append(dataprocessing.shiftedUp20(data5, dim))
		elif number == 4:
			toreturn.append(dataprocessing.shiftedDown20(data5, dim))
		elif number == 5:
			toreturn.append(dataprocessing.shiftedLeft20(data5, dim))
		elif number == 6:
			toreturn.append(dataprocessing.shiftedRight20(data5, dim))
		elif number == 7:
			toreturn.append(data5)

	toreturn = ConvertTo3DVolume(toreturn)
	try:
		proba = model.predict_proba(toreturn)
	except:
		proba = model.predict(toreturn)

	return proba


def plain_method(model_path, data):
	from keras.models import load_model
	model = load_model(model_path)

	proba = model.predict(data)
	out = np.argmax(proba, axis=1)
	return out


def average_method(model_path, data, method="average", dim=48):

	from keras.models import load_model
	model = load_model(model_path)

	proba0 = predict_prob(0, data, model, dim)
	proba1 = predict_prob(1, data, model, dim)
	proba2 = predict_prob(2, data, model, dim)
	proba3 = predict_prob(3, data, model, dim)
	proba4 = predict_prob(4, data, model, dim)
	proba5 = predict_prob(5, data, model, dim)
	proba6 = predict_prob(6, data, model, dim)
	proba7 = predict_prob(7, data, model, dim)

	from collections import Counter

	avg_out, max_out = [], []
	for row in zip(proba0, proba1, proba2, proba3, proba4, proba5, proba6, proba7):
		a = np.argmax(np.array(row).mean(axis=0))
		avg_out.append(a)


		row = np.asarray(row)
		row = np.argmax(row, axis=1)
		a = Counter(row).most_common(1)[0][0]
		max_out.append(a)
	
	avg_out = np.array(avg_out)
	max_out = np.array(max_out)
	return avg_out, max_out


def output_result_to_file(out, out_path):
	with open(out_path, 'w') as outf:
		outf.write("id,label\n")

		for cur_id, prediction in enumerate(out):
			outf.write(str(cur_id) + "," + str(prediction) + "\n")


def main():
	argc = len(sys.argv)
	if argc != 6:
		print("Usage: python test.py model_path test_path out_path plain_out val_test")
		exit()

	model_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
	method = int(sys.argv[4])
	val_test = int(sys.argv[5])

	k_fold, fold, do_zca = find_info(model_path)

	# k_fold is for testing validation data using averaging method
	# Normallt set to 1 for regular testing
	if val_test == 1:
		from keras.utils import np_utils
		#labels, data = read_data(test_path, one_hot_encoding=False)

		indice_path = "data/train_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(train, k_fold)
		else:
			indices = get_val_indices(indice_path)

		labels, data = read_data("data/train.csv", one_hot_encoding=False)
		_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])

		val_data = np.reshape(val_data, (-1, img_rows, img_cols, 1))
		val_lbl = np.asarray(val_lbl, dtype=np.uint8)

		avg_out, max_out = average_method(model_path, val_data)
		plain_out = plain_method(model_path, val_data)
		with open("record.txt", "a") as outf:
			outf.write("%s:\t\tAverage acc: %.4f, max-out acc: %.4f, normal acc: %.4f\n" % \
											(model_path,
											 (np.sum(avg_out == val_lbl) / val_lbl.shape[0]),
											 (np.sum(max_out == val_lbl) / val_lbl.shape[0]),
											 (np.sum(plain_out == val_lbl) / val_lbl.shape[0])))
	else:
		_, data = read_data(test_path, one_hot_encoding=False)

		dim = 48

		if do_zca is True:
			zca_path = os.path.join( os.path.dirname(model_path), "zca_matrix.npy")
			zca_mat = np.load(zca_path)
			data = Zerocenter_ZCA_Whitening_Global_Contrast_Normalize(data, zca_mat=zca_mat)

		data = np.asarray(data, dtype=np.float32)
		data = np.reshape(data, (-1, img_rows, img_cols))

		if method == 0:
			data = data[:, :, :, np.newaxis]
			out = plain_method(model_path, data)
		else:
			avg_out, max_out = average_method(model_path, data, dim=dim)
			if method == 1:
				print("\nMax-out...")
				out = max_out
			else:
				print("\nAveraging...")
				out = avg_out

		output_result_to_file(out, out_path)


if __name__ == "__main__":
	main()
