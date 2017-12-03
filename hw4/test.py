import os
import sys
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from data import *
from func import *

RESULT_DIR = 'results'

def main():
	model_path = sys.argv[1]
	model_name, _ = os.path.splitext(os.path.basename(model_path))

	dirpath = os.path.dirname(model_path)
	for f in os.listdir(dirpath):
		if 'vectors' in f:
			name, _ = os.path.splitext(f)
			_, top_words = name.split('_')
			top_words = int(top_words)
			wordvec_path = os.path.join(dirpath, f)


	_, word_index = load_wordvec(wordvec_path, top_words)
	tokenizer = Tokenizer()
	tokenizer.word_index = word_index

	X_test = load_test_data('data/testing_data.txt')
	X_test = tokenizer.texts_to_sequences(X_test)
	X_test = sequence.pad_sequences(X_test, maxlen=max([len(row) for row in X_test]))
	
	model = load_model(model_path)
	pred = model.predict(X_test, verbose=1)
	pred = np.around(pred)
	write_to_file(pred, RESULT_DIR + "/" + model_name + ".csv")


if __name__ == '__main__':
	main()
