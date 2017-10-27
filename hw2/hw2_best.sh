#!/bin/bash
python3 dataProcessing.py $2 data/my_test test
python3 bset/test.py data/my_test result/10.csv bset/submit/10.gbdt/20-18 gbdt
python3 bset/test.py data/my_test result/13.csv bset/submit/13.d5_n500_soft_mlog_1_1/20-18 gbdt
python3 bset/test.py data/my_test result/14.csv bset/submit/14.tuned_models/20-8 gbdt_wrapper
python3 ensemble.py result/10.csv result/13.csv result/14.csv $6
