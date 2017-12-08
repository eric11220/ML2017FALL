import os
import sys
import _pickle as pk
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from data import *
from func import *

RESULT_DIR = 'results'

def main():
	model_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]

	model_name, _ = os.path.splitext(os.path.basename(model_path))

	dirpath = os.path.dirname(model_path)
	tokenizer = pk.load(open(os.path.join(dirpath, 'tokenizer.pk'), 'rb'))

	X_test = load_test_data(test_path)
	X_test = tokenizer.texts_to_sequences(X_test)
	X_test = sequence.pad_sequences(X_test, maxlen=max([len(row) for row in X_test]))
	
	model = load_model(model_path)
	print(model.summary())
	pred = model.predict(X_test, verbose=1)
	pred = np.around(pred)
	write_to_file(pred, out_path)


if __name__ == '__main__':
	main()
