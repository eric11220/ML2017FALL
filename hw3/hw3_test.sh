#/bin/bash

# Download three models
wget https://www.dropbox.com/s/b2zpucuuva7bj4n/Model.211-0.6870-0.9480.hdf5?dl=0
wget https://www.dropbox.com/s/czyeh7ddocprsyo/Model.253-0.7031-0.9231.hdf5?dl=0 
wget https://www.dropbox.com/s/xc8kdgniu5p4l6f/Model.419-0.6871-1.6949.hdf5?dl=0

# Run each model
python3 test.py Model.211-0.6870-0.9480.hdf5 $1 1.csv 0 0
python3 test.py Model.253-0.7031-0.9231.hdf5 $1 2.csv 0 0
python3 test.py Model.419-0.6871-1.6949.hdf5 $1 3.csv 0 0

python3 ensemble.py 1.csv 2.csv 3.csv $2

