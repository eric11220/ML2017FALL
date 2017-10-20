import os
import sys
import tensorflow as tf 
from splitData import *

model_dir = None
dout = 0.5
limit_inc_loss = 5
best_loss = 10000
layers =[64]

def network(train_data, train_label, val_data, val_label, k_fold, fold, layers, dropouts, train_mean, train_std, feat_order, n_epoch=10000, lr=1, batch_size=1, display_epoch=10):
	global model_dir, best_loss
	model_path = os.path.join(model_dir, 'model.ckpt')

	layers.insert(0, train_data.shape[1])
	layers.append(train_label.shape[1])

	num_data = train_data.shape[0]

	# Construct neural network
	x = tf.placeholder(tf.float32, [None, layers[0]], name="x")
	y = tf.placeholder(tf.float32, [None, train_label.shape[1]], name="y")

	prob = tf.placeholder(tf.float32, name="dropout") 

	layer = x
	for layer_id, prev_n_neuron, n_neuron, dropout in zip(range(len(layers)), layers[:-1], layers[1:], dropouts[1:]):
		w = tf.Variable(tf.truncated_normal([prev_n_neuron, n_neuron], stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[n_neuron]))
		layer = tf.add(tf.matmul(layer, w), b)
		layer = tf.nn.relu(layer)

		if dropout is True:
			layer = tf.nn.dropout(layer, prob)
	
	first_y = tf.slice(y, [0, 0], [tf.shape(y)[0], 1])
	first_logits = tf.slice(layer, [0, 0], [tf.shape(layer)[0], 1], name="predict")
	print_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(first_y, first_logits)))) 

	#weights = [0.7 ** p for p in range(val_label.shape[1])]
	loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, layer)))) 
	optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(loss)

	# Start training
	num_inc_losses, prev_loss = 0, 0
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epoch):
			batch_idx, iter = 0, 1

			indices = np.random.permutation(num_data) 
			train_data = train_data[indices]
			train_label = train_label[indices]

			while batch_idx < num_data:
				batch_data = train_data[batch_idx:batch_idx+batch_size, :]
				batch_label = train_label[batch_idx:batch_idx+batch_size]
				_, l = sess.run(
					[optimizer, print_loss],
					feed_dict={
						x: batch_data,
						y: batch_label,
						prob: dout
					}
				)

				batch_idx += batch_size
				iter += 1

			if epoch % display_epoch == 0 or epoch == n_epoch-1:
				_, train_loss = sess.run(
					[optimizer, print_loss],
					feed_dict={
						x: train_data,
						y: train_label,
						prob: 1
					}
				)

				if val_data is not None:
					_, val_loss, p = sess.run(
						[optimizer, print_loss, first_logits],
						feed_dict={
							x: val_data,
							y: val_label,
							prob: 1
						}
					)
					if val_loss > prev_loss:
						num_inc_losses += 1
					else:
						num_inc_losses = 0
					prev_loss = val_loss

					if val_loss < best_loss:
						print("validation loss %f better than best loss %f, saving the model..." % (val_loss, best_loss))
						saver.save(sess, model_path)
						train_info_path = os.path.join(model_dir, "train_info.npy")
						np.save(train_info_path, [train_mean, train_std, feat_order])
						best_loss = val_loss

					if num_inc_losses > limit_inc_loss:
						print("Validation loss keeps increasing for %d times, early stopping!" % limit_inc_loss)
						break

					print('>epoch=%d, lrate=%.5f, error=%.3f, validation error=%.6f, num_inc_losses=%d' % (epoch, lr, train_loss, val_loss, num_inc_losses))
				else:
					print('>epoch=%d, lrate=%.5f, error=%.3f' % (epoch, lr, train_loss))

		if val_data is None:
			saver.save(sess, model_path)
			train_info_path = os.path.join(model_dir, "train_info.npy")
			np.save(train_info_path, [train_mean, train_std, feat_order])

	if val_data is not None:
		return val_loss
	else:
		return None

def main():
	argc = len(sys.argv)
	if argc != 6:
		print("Usage: python linearReg.py input_csv k_fold n_epoch lr batch_size")
		exit()

	global model_dir 

	in_csv = sys.argv[1]
	k_fold = int(sys.argv[2])
	n_epoch = int(sys.argv[3])
	lr = float(sys.argv[4])
	batch_size = int(sys.argv[5])

	dropouts = [False, True, False] 

	model_no = "_".join([str(layer) for layer in layers])
	model_dir = os.path.join('model', model_no + '_dropout' + str(dout))
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	name, ext = os.path.splitext(in_csv)
	next_hour = name.split("+")
	if len(next_hour) == 1:
		label_len = 1
	else:
		label_len = int(next_hour[1])

	data_dir = os.path.dirname(in_csv)
	first_fold_path = os.path.join(data_dir, "indices+" + str(label_len) + "_" + str(k_fold) + "_0")
	if not os.path.isfile(first_fold_path):
		gen_val_indices(in_csv, k_fold)

	sum_error = 0
	for i in range(k_fold):
		fold_path = os.path.join(data_dir, "indices" + str(k_fold) + "_" + str(i))
		train_data, val_data, train_lbl, val_lbl, feat_order = split_data(in_csv, fold_path)

		train_lbl = np.reshape(train_lbl, [-1, 1])
		val_lbl = np.reshape(val_lbl, [-1, 1])

		train_mean = np.mean(train_data, axis=0)
		train_std = np.std(train_data, axis=0)
		train_data = (train_data - train_mean) / train_std

		if val_data is not None:
			val_data = (val_data - train_mean) / train_std

		#train_info_path = os.path.join(model_dir, str(k_fold) + "-" + str(i) + "_train_info.npy")
		error = network(train_data, train_lbl, val_data, val_lbl, k_fold, i, layers, dropouts, train_mean, train_std, feat_order, batch_size=batch_size, n_epoch=n_epoch, lr=lr)

		if error is not None:
			sum_error += error

	if  k_fold > 1:
		print(sum_error / k_fold)

if __name__ == '__main__':
	main()
