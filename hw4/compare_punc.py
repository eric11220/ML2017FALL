import os

'''
for top_words in [15000, 20000, 25000]:
	for dropout in [0.0, 0.1, 0.2, 0.3]:
		modeldir = "100_top%d_dropout%f" % (top_words, dropout)
		os.system("python3 train.py --wordvec data/word_vec/all_no_punc_100/vectors.txt --dropout %f --modeldir models/%s" % (dropout, modeldir))
'''

for top_words in [15000, 20000, 25000]:
	for dropout in [0.0, 0.1, 0.2, 0.3]:
		modeldir = "200_top%d_dropout%f" % (top_words, dropout)
		os.system("python3 train.py --wordvec data/word_vec/all_no_punc_200/vectors.txt --dropout %f --modeldir models/%s" % (dropout, modeldir))
