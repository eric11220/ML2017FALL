#!/bin/bash
python3 train.py $1 --trainable false --gensim wordvec/gensim_all_256/model --dropout 0.5 --top_words 55000
