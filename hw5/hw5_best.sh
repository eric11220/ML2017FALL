#!/bin/bash
python3 test.py submits/ensemble/Model.22-0.7599.hdf5 $1 1.csv
python3 test.py submits/ensemble/Model.33-0.7442.hdf5 $1 2.csv
python3 test.py submits/ensemble/Model.34-0.7721.hdf5 $1 3.csv
python3 ensemble.py 1.csv 2.csv 3.csv $2
