from __future__ import print_function

import sys

import dataprocessing
from splitData import *

img_rows, img_cols, nb_classes = 48, 48, 7

def ConvertTo3DVolume(data):
	data = np.asarray(data) 
	data = np.reshape(data, [-1, img_rows, img_cols, 1])
	data = data.astype('float32')

	return data


def ImageTransform(number, data, dim):
	toreturn = []
	for dat in data:
		if number == 0:
			toreturn.append(dataprocessing.Flip(dat, dim))
		elif number == 1:
			toreturn.append(dataprocessing.Roated15Left(dat, dim))
		elif number == 2:
			toreturn.append(dataprocessing.Roated15Right(dat, dim))
		elif number == 3:
			toreturn.append(dataprocessing.shiftedUp20(dat, dim))
		elif number == 4:
			toreturn.append(dataprocessing.shiftedDown20(dat, dim))
		elif number == 5:
			toreturn.append(dataprocessing.shiftedLeft20(dat, dim))
		elif number == 6:
			toreturn.append(dataprocessing.shiftedRight20(dat, dim))
		elif number == 7:
			toreturn.append(dat)

	return ConvertTo3DVolume(toreturn)


def load_models(model_dir, model1_path, model2_path):
	from keras.models import load_model
	model1 = load_model(model1_path)
	model2 = load_model(model2_path)

	regress_models = []
	for cls in range(nb_classes):
		for name in os.listdir(model_dir):
			if "Model" + str(cls) in name:
				model_path = os.path.join(model_dir, name)
				model = load_model(model_path)
				regress_models.append(model)

	return model1, model2, regress_models


def plain_method(model_dir, model1, model2, regress_models, data, prob=False):
	nn_feat1 = nn_feature(model1, data)
	nn_feat2 = nn_feature(model2, data)

	# Concate two nn featues
	final_feat = np.concatenate((nn_feat1, nn_feat2), axis=1)

	# Get a score from each regression model
	scores = []
	for cls in range(nb_classes):
		print("Processing class %d..." % cls)
		num_data = len(final_feat)
		idx, batch, tmp_score = 0, 1000, None
		while(idx < num_data):
			if idx + batch > num_data:
				end = num_data
			else:
				end = idx + batch

			model = regress_models[cls]

			result = model.predict(final_feat[idx:end])
			result = np.reshape(result, (-1))
			if tmp_score is None:
				tmp_score = result
			else:
				tmp_score = np.concatenate((tmp_score, result))

			idx += batch

		scores.append(tmp_score)

	scores = np.asarray(scores)
	scores = np.reshape(scores, (scores.shape[0], -1))
	if prob is False:
		prediction = np.argmax(scores, axis=0)
	else:
		prediction = scores
	return prediction


def average_method(model_dir, model1, model2, regress_models, data, dim=48):
	from keras.models import load_model

	out = []
	for form in range(7):
		print("Processing transformed image %d..." % form)
		trans_data = ImageTransform(form, data, dim)
		prediction = plain_method(model_dir, model1, model2, regress_models, trans_data, prob=True)
		out.append(prediction)

	out = np.asarray(out)

	# "average" method
	mean_prob = out.mean(axis=0)
	average = np.argmax(mean_prob, axis=0)

	# "max_vote" method
	from collections import Counter

	votes = np.argmax(out, axis=1)
	votes = np.transpose(votes)

	result = []
	for row in votes:
		a = Counter(row).most_common(1)[0][0]
		result.append(a)

	max_vote = np.asarray(result)

	return average, max_vote


def output_result_to_file(out, out_path):
	with open(out_path, 'w') as outf:
		outf.write("id,label\n")

		for cur_id, prediction in enumerate(out):
			outf.write(str(cur_id) + "," + str(prediction) + "\n")


def main():
	argc = len(sys.argv)
	if argc != 6:
		print("Usage: python test.py model_dir test_path out_path method val_test")
		exit()

	# model_dir is directory
	model_dir = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
	method = int(sys.argv[4])
	val_test = int(sys.argv[5])

	# Get two model paths
	tmp_path = os.path.basename(os.path.normpath(model_dir))
	model1_path, model2_path, fold_info = tmp_path.split('_')
	model1_path = os.path.join(model_dir, model1_path + '.hdf5')
	model2_path = os.path.join(model_dir, model2_path + '.hdf5')

	# Preload Models
	model1, model2, regress_models = load_models(model_dir, model1_path, model2_path)

	# k_fold is for testing validation data using averaging method
	# Normallt set to 1 for regular testing
	if val_test == 1:
		test_path = "data/train.csv"

		k_fold, fold = fold_info.split('-')
		k_fold, fold = int(k_fold), int(fold)
		indice_path = "data/train_" + str(k_fold) + "fold"
		if not os.path.isfile(indice_path):
			print("Fold combination not exists!")
			exit()
		else:
			indices = get_val_indices(indice_path)

		labels, data = read_data(test_path, one_hot_encoding=False)
		_, val_data, _, val_lbl = split_train_val(data, labels, indices[fold])

		val_data = np.reshape(val_data, (-1, img_rows, img_cols, 1))
		if method == 0:
			plain_out = plain_method(model_dir, model1, model2, regress_models, val_data)
			print("Normal method acc: %.4f" % (np.sum(plain_out == val_lbl) / val_lbl.shape[0]))
		else:
			avg_out, max_out = average_method(model_dir, model1, model2, regress_models, val_data, dim=48)
			print("Average method acc: %.4f" % (np.sum(avg_out == val_lbl) / val_lbl.shape[0]))
			print("Max-vote method acc: %.4f" % (np.sum(max_out== val_lbl) / val_lbl.shape[0]))
	else:
		_, data = read_data(test_path, one_hot_encoding=False)

		'''
		if do_zca is True:
			zca_path = os.path.join( os.path.dirname(model_dir), "zca_matrix.npy")
			zca_mat = np.load(zca_path)
			data = Zerocenter_ZCA_Whitening_Global_Contrast_Normalize(data, zca_mat=zca_mat)
		'''

		data = np.reshape(data, (-1, img_rows, img_cols, 1))
		if method == 0:
			out = plain_method(model_dir, model1, model2, regress_models, data)
		else:
			print("Averaging...")
			avg_out, max_out = average_method(model_dir, model1, model2, regress_models, data, dim=48)
			
			if method == 1:
				out = max_out
			else:
				out = avg_out

		output_result_to_file(out, out_path)


if __name__ == "__main__":
	main()
