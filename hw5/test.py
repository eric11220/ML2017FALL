import sys
from read_data import *
from keras.models import load_model


def main():
	model_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]

	model = load_model(model_path)
	test_data = read_test_data(test_path)

	uid, mid = test_data[:, 0], test_data[:, 1]
	pred = model.predict([uid, mid], verbose=1)

	'''
	_, _, _, _, ratings_mean, ratings_std = read_train_data("data/train_shuf_10fold.csv", 1)
	pred = (pred * ratings_std) + ratings_mean
	'''
	write_result_to_file(pred, out_path)


if __name__ == '__main__':
	main()
