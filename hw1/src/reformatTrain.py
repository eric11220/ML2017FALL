import os
import sys
import codecs
import numpy as np
from keep_feat import *

DATA_DIR = "../data"
train_csv = os.path.join(DATA_DIR, "train.csv")
target = 'PM2.5'

def padd_missing_values(feats):
	for key, values in feats.items():
		pass

# Reformat original training csv into per row data
def write_feats_to_file(month, out_path, feats, prev_hour, next_hour, feat_names):
	global keep_feats

	if len(feats) == 0:
		return

	with open(out_path, 'a') as outf:
		start_idx, num_hour = 0, len(list(feats.items())[0][1])

		# Every #prev_hour sample a datum and write to output file
		while start_idx + prev_hour + next_hour <= num_hour:
			final_feat, labels = [], 0
			for key in keep_feats:
				feat_list = feats[key]
				if key == target:
					labels = feat_list[start_idx + prev_hour:start_idx + prev_hour + next_hour]

				final_feat.extend(feat_list[start_idx:start_idx + prev_hour])

			if "-1" in labels or "-1" in final_feat:
				pass
			else:
				outf.write(str(month) + " ")
				for feat in final_feat:
					# Change 'NR' to 0
					try:
						_ = float(feat)
					except:
						feat = "0"
					outf.write(feat + " ")
				for label in labels:
					outf.write(label + " ")
				outf.write("\n")

			start_idx += 1

# Transform csv into rows of training data
def parse_train_csv(csv_path , prev_hour=9, next_hour=1):

	# Name of output folder is based on number of previous hour
	out_dir = os.path.join(DATA_DIR, "num_prev_hour" + str(prev_hour))
	if not os.path.isdir(out_dir):
		os.mkdir(out_dir)

	#out_path = os.path.join(out_dir, "train_dat+" + str(next_hour) + "_" + keep_feats[0] + ".csv")
	out_path = os.path.join(out_dir, "train_dat+" + str(next_hour) + ".csv")

	# Truncate existing file in case of data duplication
	if os.path.isfile(out_path):
		with open(out_path, 'w') as inf:
			pass

	cur_month, writen_header = None, False
	feats, feat_names = {}, []
	with codecs.open(csv_path, 'r', encoding='utf-8', errors='ignore') as inf:
		inf.readline()
		for line in inf:
			vals = line.strip().split(',')
			date, feat_name, feat = vals[0], vals[2], vals[3:]
			_, m, _ = date.split("/")

			# Concat to the end of that feature list
			if cur_month == m or cur_month is None:
				if cur_month is None:
					cur_month = m

				if feat_name not in feat_names:
					feat_names.append(feat_name)
				if feats.get(feat_name, None) is None:
					feats[feat_name] = feat
				else:
					feats[feat_name].extend(feat)
			# A new month, stop concatenation
			else:
				if writen_header is False:
					# Write feature order in the first line
					with open(out_path, 'a') as outf:
						for key in keep_feats:
							outf.write(key + " ")
						outf.write("\n")
					writen_header = True

				write_feats_to_file(cur_month, out_path, feats, prev_hour, next_hour, feat_names)
				feats = {feat_name: feat}
				cur_month = m

	# Last month
	write_feats_to_file(cur_month, out_path, feats, prev_hour, next_hour, feat_names)

def main():
	global keep_feats

	prev_hour = int(sys.argv[1])
	next_hour = int(sys.argv[2])
	if len(sys.argv) == 4:
		random_pick_feat_n = int(sys.argv[3])
		keep_indices = np.random.choice(range(len(keep_feats)), random_pick_feat_n, replace=False)
		keep_feats = np.asarray(keep_feats)[keep_indices]
		keep_feats = keep_feats.tolist()

	print(keep_feats, len(keep_feats))

	'''
	whole_feats = keep_feats
	for i in range(len(keep_feats)):
		keep_feats = whole_feats
		keep_feats = keep_feats[i:i+1]
	'''

	parse_train_csv(train_csv, prev_hour, next_hour)

if __name__ == '__main__':
	main()
