import argparse
from read_data import *
from sklearn.model_selection import StratifiedKFold


latent_dim = 200
epochs = 50
batch_size = 64


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_data", help="Train data path")

	parser.add_argument("--kfold", help="K-fold", default=1, type=int)
	parser.add_argument("--model_dir", help="Model location")
	parser.add_argument("--method", help="Training method", default="mf", choices=["mf", "nn"])
	return parser.parse_args()


def mf(num_user, num_movie, latent_dim):
	from keras.models import Model
	from keras.layers import dot, add, Input, Flatten
	from keras.layers.embeddings import Embedding

	user = Input(shape=[1])
	movie = Input(shape=[1])

	user_weight = Flatten()(Embedding(num_user, latent_dim)(user))
	user_bias = Flatten()(Embedding(num_user, 1)(user))

	movie_weight = Flatten()(Embedding(num_movie, latent_dim)(movie))
	movie_bias = Flatten()(Embedding(num_movie, 1)(movie))

	dot_result = dot([user_weight, movie_weight], axes=-1)
	added_user_bias = add([dot_result, user_bias])
	added_movie_bias = add([added_user_bias, movie_bias])

	model = Model(inputs=[user, movie], outputs=added_movie_bias)
	model.compile('adam', 'mean_squared_error', metrics=["accuracy"])
	model.summary()
	return model


def compute_accuracy(model, x, y):
	yhat = model.predict([x[:, 0], x[:, 1]])
	yhat[yhat > 5.0] = 5.0
	yhat[yhat < 0.0] = 0.0
	yhat = np.around(yhat)
	yhat = np.reshape(yhat, -1)
	acc = np.sum(yhat.astype(int) == y.astype(int)) / len(y)
	return acc


def main():
	args = parse_input()
	user_movie, ratings, num_user, num_movie = read_train_data(args.train_data)

	if args.kfold > 1:
		skf = StratifiedKFold(n_splits=args.kfold)
		for cv_idx, (train_index, val_index) in enumerate(skf.split(user_movie, ratings)):
			train_x, train_y  = user_movie[train_index], ratings[train_index]
			val_x, val_y = user_movie[val_index], ratings[val_index]

			model = mf(num_user, num_movie, latent_dim)
			model.fit([train_x[:, 0], train_x[:, 1]], 
								train_y,
								epochs=epochs,
								validation_data=([val_x[:, 0], val_x[:, 1]], val_y),
								batch_size=batch_size)
	else:
		model = mf(num_user, num_movie, latent_dim)
		model.fit(user_movie, ratings)


if __name__ == '__main__':
	main()
