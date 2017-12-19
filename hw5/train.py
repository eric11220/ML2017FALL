import os
import argparse
from read_data import *
from model import *
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint


MODEL_DIR = "models"
USER_INFO_VEC_PATH = "data/user_vector.csv"
MOVIE_INFO_VEC_PATH = "data/movie_vector.csv"
epochs = 100
batch_size = 64


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_data", help="Train data path", default="data/train_shuf_10fold.csv")

	parser.add_argument("--kfold", help="K-fold", default=10, type=int)
	parser.add_argument("--latent_dim", help="latent dimension", default=200, type=int)
	parser.add_argument("--dropout", help="Dropout rate", default=0.5, type=float)
	parser.add_argument("--method", help="Training method", default="nn", choices=["mf", "nn", "classification"])
	parser.add_argument("--modeldir", help="Model location")
	return parser.parse_args()


def main():
	args = parse_input()
	user_movie, ratings, num_user, num_movie = read_train_data(args.train_data)

	if args.modeldir is not None:
		modeldir = args.modeldir
	else:
		time_now = datetime.now().strftime('%m-%d-%H-%M')
		modeldir = os.path.join(MODEL_DIR, "%s_hid%d_dropout%.1f_%s" % (args.method, args.latent_dim, args.dropout, time_now))
	os.makedirs(modeldir, exist_ok=True)

	if args.kfold > 1:
		skf = StratifiedKFold(n_splits=args.kfold)
		splits = skf.split(user_movie, ratings)

		if args.method == "classification":
			from keras.utils import to_categorical
			ratings = ratings - 1
			ratings = to_categorical(ratings)

		for cv_idx, (train_index, val_index) in enumerate(splits):
			if cv_idx == 0:
				continue
			sub_modeldir = os.path.join(modeldir, "%d-%d" % (args.kfold, cv_idx))
			os.makedirs(sub_modeldir, exist_ok=True)

			train_x, train_y  = user_movie[train_index], ratings[train_index]
			val_x, val_y = user_movie[val_index], ratings[val_index]

			filepath = os.path.join(sub_modeldir, 'Model.{epoch:02d}-{val_loss:.4f}.hdf5')
			ckpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=10, save_best_only=True, mode='auto')

			if args.method == 'mf':
				model = mf(num_user, num_movie, args.latent_dim, args.dropout)
				model.fit([train_x[:, 0], train_x[:, 1]],
									train_y,
									epochs=epochs,
									validation_data=([val_x[:, 0], val_x[:, 1]], val_y),
									batch_size=batch_size,
									callbacks=[ckpointer])
			elif args.method == 'nn' or args.method == 'classification':
				user_info = read_info("data/user_vector.csv")
				movie_info = read_info("data/movie_vector.csv")
				if args.method == 'nn':
					model = nn(args.latent_dim, args.dropout, user_info, movie_info)
					model.fit([train_x[:, 0], train_x[:, 1]],
										train_y,
										epochs=epochs,
										validation_data=([val_x[:, 0], val_x[:, 1]], val_y),
										batch_size=batch_size,
										callbacks=[ckpointer])
				else:
					model = nn(args.latent_dim, args.dropout, user_info, movie_info, classification=True)
					model.fit([train_x[:, 0], train_x[:, 1]],
										train_y,
										epochs=epochs,
										validation_data=([val_x[:, 0], val_x[:, 1]], val_y),
										batch_size=batch_size,
										callbacks=[ckpointer])
	else:
		model = mf(num_user, num_movie, args.latent_dim, args.dropout)
		model.fit(user_movie, ratings)


if __name__ == '__main__':
	main()
