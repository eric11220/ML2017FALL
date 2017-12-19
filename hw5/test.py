import sys
from read_data import read_test_data, write_result_to_file
from keras.models import load_model


def main():
	model_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]

	model = load_model(model_path)
	test_data = read_test_data(test_path)

	uid, mid = test_data[:, 0], test_data[:, 1]
	pred = model.predict([uid, mid], verbose=1)

	write_result_to_file(pred, out_path)


if __name__ == '__main__':
	main()
