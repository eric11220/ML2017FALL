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
	plain_out = sys.argv[4] == "1"
	val_test = int(sys.argv[5])


	k_fold, fold, do_zca = find_info(test_path)

	# k_fold is for testing validation data using averaging method
	# Normallt set to 1 for regular testing
	if val_test == 1:
		from keras.utils import np_utils
		#labels, data = read_data(test_path, one_hot_encoding=False)

		indice_path = name + "_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			indices = gen_val_indices(train, k_fold)
		else:
			indices = get_val_indices(indice_path)

		labels, data = read_data(test_path, one_hot_encoding=False)
		_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])

		val_data = val_data[:, :, :, np.newaxis]
		val_lbl = np.asarray(val_lbl, dtype=np.uint8)

		avg_out = average_method(model_path, val_data)
		print("Average method acc: %.4f" % (np.sum(avg_out == val_lbl) / val_lbl.shape[0]))

		val_data = np.reshape(val_data, (-1, img_rows, img_cols, 1))
		plain_out = plain_method(model_path, val_data)
		print("Normal method acc: %.4f" % (np.sum(plain_out == val_lbl) / val_lbl.shape[0]))
	else:
		_, data = read_data(test_path, one_hot_encoding=False)
		if do_zca is True:
			zca_path = os.path.join( os.path.dirname(model_path), "zca_matrix.npy")
			zca_mat = np.load(zca_path)
			data = Zerocenter_ZCA_Whitening_Global_Contrast_Normalize(data, zca_mat=zca_mat)

		data = np.asarray(data, dtype=np.float32)
		data = np.reshape(data, (-1, img_rows, img_cols))

		if plain_out is True:
			data = data[:, :, :, np.newaxis]
			out = plain_method(model_path, data)
		else:
			print("Averaging...")
			out = average_method(model_path, data)
		output_result_to_file(out, out_path)


if __name__ == "__main__":
	main()
