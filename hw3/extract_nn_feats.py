import sys
from splitData import *

img_rows, img_cols, nb_classes = 48, 48, 7
concat_dir = "concat_feats"
MODEL_DIR = "models"

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python extract_nn_feats.py model_in1 model_in2")
		exit()

	model_in1 = sys.argv[1]
	model_in2 = sys.argv[2]
				
	model1, _ = os.path.splitext(os.path.basename(model_in1))
	model2, _ = os.path.splitext(os.path.basename(model_in2))

	k_fold1, fold1, _ = find_info(model_in1)
	k_fold2, fold2, _ = find_info(model_in2)

	feat_file = os.path.join(concat_dir, model1 + '_' + model2 + '_' + str(k_fold1) + '-' + str(fold1) + '.csv')
	if os.path.isfile(feat_file):
		os.remove(feat_file)

	'''
	if k_fold1 != k_fold2 or fold1 != fold2:
		print("Fold not match, please check models")
		exit()
	'''

	train_csv_name = 'data/train.csv'
	name, ext = os.path.splitext(train_csv_name)

	labels, data = read_data(train_csv_name, one_hot_encoding=False)

	data = np.asarray(data) 
	data = data.reshape(data.shape[0], img_rows, img_cols)
	data = data.reshape(data.shape[0], img_rows, img_cols, 1)
	data = data.astype('float32')
	
	from keras.models import load_model
	model1 = load_model(model_in1)
	model2 = load_model(model_in2)

	# Copy model to feat_ensemble
	model_subdir = os.path.join(MODEL_DIR, 'feat_ensemble', os.path.splitext(os.path.basename(feat_file))[0])
	if not os.path.isdir(model_subdir):
		os.makedirs(model_subdir)

	from shutil import copyfile
	copyfile(model_in1, os.path.join(model_subdir, os.path.basename(model_in1)))
	copyfile(model_in2, os.path.join(model_subdir, os.path.basename(model_in2)))
	
	num_data = len(data)
	idx, batch = 0, 100
	while idx < num_data:
		if idx + batch > num_data:
			end = num_data
		else:
			end = idx + batch

		print("Processing training data idx %d to %d..." % (idx, end))

		model_feat1 = nn_feature(model1, data[idx:end, :, :, :])
		model_feat2 = nn_feature(model2, data[idx:end, :, :, :])
		
		final_feat = np.concatenate((model_feat1, model_feat2), axis=1)
		append_to_file(feat_file, final_feat, labels[idx:end] , idx)
		idx = end


def append_to_file(feat_file, feats, labels, idx):
	with open(feat_file, "a") as outf:
		for feat, label in zip(feats, labels):
			outf.write(str(label) + ',')
			for val in feat:
				outf.write(str(val) + ' ')
			outf.write('\n')


if __name__ == '__main__':
	main()
