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


def predict_prob(number, data, model):
	toreturn = []
	for data5 in data:
		if number == 0:
			toreturn.append(dataprocessing.Flip(data5))
		elif number == 1:
			toreturn.append(dataprocessing.Roated15Left(data5))
		elif number == 2:
			toreturn.append(dataprocessing.Roated15Right(data5))
		elif number == 3:
			toreturn.append(dataprocessing.shiftedUp20(data5))
		elif number == 4:
			toreturn.append(dataprocessing.shiftedDown20(data5))
		elif number == 5:
			toreturn.append(dataprocessing.shiftedLeft20(data5))
		elif number == 6:
			toreturn.append(dataprocessing.shiftedRight20(data5))
		elif number == 7:
			toreturn.append(data5)

	toreturn = ConvertTo3DVolume(toreturn)
	proba = model.predict_proba(toreturn)
	return proba


def plain_method(model_path, data):
	from keras.models import load_model
	model = load_model(model_path)

	proba = model.predict(data)
	out = np.argmax(proba, axis=1)
	return out


def average_method(model_path, data, method="average"):

	from keras.models import load_model
	model = load_model(model_path)

	proba0 = predict_prob(0, data, model)
	proba1 = predict_prob(1, data, model)
	proba2 = predict_prob(2, data, model)
	proba3 = predict_prob(3, data, model)
	proba4 = predict_prob(4, data, model)
	proba5 = predict_prob(5, data, model)
	proba6 = predict_prob(6, data, model)
	proba7 = predict_prob(7, data, model)

	out = []
	for row in zip(proba0, proba1, proba2, proba3, proba4, proba5, proba6, proba7):
		# Row is of size (#transfored_imgs, #n_classes)
		if method == "average":
			a = np.argmax(np.array(row).mean(axis=0))
			out.append(a)
		elif method == "max_vote":
			from collections import Counter
			row = np.asarray(row)
			row = np.argmax(row, axis=1)
			a = Counter(row).most_common(1)[0][0]
			out.append(a)
	
	out = np.array(out)
	return out


def main():
	argc = len(sys.argv)
	if argc != 4:
		print("Usage: python test.py model_path test_path val_test")
		exit()

	model_path = sys.argv[1]
	test_path = sys.argv[2]
	val_test = int(sys.argv[3])


	# k_fold is for testing validation data using averaging method
	# Normallt set to 1 for regular testing
	if val_test == 1:
		from keras.utils import np_utils
		#labels, data = read_data(test_path, one_hot_encoding=False)

		name, ext = os.path.splitext(test_path)
		fold_info = model_path.split('/')[-2]
		k_fold, fold = fold_info.split('-')
		k_fold, fold  = int(k_fold), int(fold)

		indice_path = name + "_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(train, k_fold)
		else:
			indices = get_val_indices(indice_path)

		labels, data = read_data(test_path, one_hot_encoding=False)
		_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])
		Val_x, Val_y = dataprocessing.convert_data(val_data, val_lbl)

		Val_x = np.asarray(Val_x)
		Val_x = Val_x[:, :, :, np.newaxis]
		Val_x = Val_x.astype('float32')
		Val_y = np.asarray(Val_y, dtype=np.uint8)

		avg_out = average_method(model_path, Val_x)
		print("Average method acc: %.4f" % (np.sum(avg_out == Val_y) / Val_y.shape[0]))

		plain_out = plain_method(model_path, Val_x)
		print("Normal method acc: %.4f" % (np.sum(plain_out == Val_y) / Val_y.shape[0]))
	else:
		data = dataprocessing.load_test_data()
		average_method(model_path, data)


if __name__ == "__main__":
	main()
