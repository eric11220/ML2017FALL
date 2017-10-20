import os
import sys
import tensorflow as tf 
from splitData import *

def predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, model_path):

	model_dir = os.path.dirname(model_path)

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_path)
		saver.restore(sess,tf.train.latest_checkpoint(model_dir))

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("x:0")
		dropout = graph.get_tensor_by_name("dropout:0")
		predict = graph.get_tensor_by_name("predict:0")

		with open(out_csv, 'w') as outf:
			outf.write("id,value\n")
			with open(in_csv, 'r') as inf:
				cur_id, all_feats = None, {}
				for line in inf:
					vals = line.strip().split(',')
					id, feat_name, feat = vals[0], vals[1], vals[2:]

					for idx, f in enumerate(feat):
						try:
							_ = float(f)
						except:
							feat[idx] = 0

					if cur_id == None or cur_id == id:
						all_feats[feat_name] = feat
						cur_id = id
					else:
						final_feat = []
						for name in feat_order:
							final_feat.extend(all_feats[name])
					
						final_feat = np.asarray(final_feat, dtype=np.float32)
						final_feat = (final_feat - train_mean) / train_std
						final_feat = np.reshape(final_feat, [1, -1])

						prediction = sess.run(
							predict,
							feed_dict = {
								x: final_feat,
								dropout: 1
							}
						)

						outf.write(cur_id + "," + str(prediction[0][0]) + "\n")
						cur_id, all_feats = id, {feat_name: feat}

			final_feat = []
			for name in feat_order:
				final_feat.extend(all_feats[name])
			final_feat = np.asarray(final_feat, dtype=np.float32)

			final_feat = (final_feat - train_mean) / train_std
			final_feat = np.reshape(final_feat, [1, -1])

			prediction = sess.run(
				predict,
				feed_dict = {
					x: final_feat,
					dropout: 1
				}
			)
			outf.write(cur_id + "," + str(prediction[0][0]) + "\n")

def main():
	argc = len(sys.argv)
	if argc != 4:
		print("Usage: python test.py test_csv out_csv model_path")
		exit()

	in_csv = sys.argv[1]
	out_csv = sys.argv[2]
	model_path = sys.argv[3]

	model_dir = os.path.dirname(model_path)
	train_info_path = os.path.join(model_dir, "train_info.npy")

	train_mean, train_std, feat_order = np.load(train_info_path)

	name, ext = os.path.splitext(in_csv)
	next_hour = name.split("+")
	if len(next_hour) == 1:
		label_len = 1
	else:
		label_len = int(next_hour[1])

	predict_csv(in_csv, out_csv, feat_order, train_mean, train_std, model_path)

if __name__ == '__main__':
	main()
