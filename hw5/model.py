from keras.models import Model, Sequential
from keras.layers import dot, add, Concatenate 
from keras.layers import  Input, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding

def embedding(num_user, num_movie, latent_dim, dropout):
	user = Input(shape=[1])
	movie = Input(shape=[1])

	user_weight = Flatten()(Embedding(num_user, latent_dim)(user))
	user_weight = Dropout(dropout)(user_weight)

	movie_weight = Flatten()(Embedding(num_movie, latent_dim)(movie))
	movie_weight = Dropout(dropout)(movie_weight)

	movie_bias = Flatten()(Embedding(num_movie, 1)(movie))
	user_bias = Flatten()(Embedding(num_user, 1)(user))
	return user, movie, user_weight, user_bias, movie_weight, movie_bias


def mf(num_user, num_movie, latent_dim, dropout):
	user, movie, user_weight, user_bias, movie_weight, movie_bias = embedding(num_user, num_movie, latent_dim, dropout)

	dot_result = dot([user_weight, movie_weight], axes=-1)
	added_user_bias = add([dot_result, user_bias])
	dot_result = add([added_user_bias, movie_bias])

	model = Model(inputs=[user, movie], outputs=dot_result)
	model.compile('adam', 'mean_squared_error', metrics=["accuracy"])
	model.summary()
	return model


def nn(latent_dim, dropout, user_info, movie_info, classification=False):
	from keras import regularizers
	num_user, user_info_len = user_info.shape
	num_movie, movie_info_len = movie_info.shape

	# Latent representation
	user, movie, user_weight, _, movie_weight, _ = embedding(num_user, num_movie, latent_dim, dropout)

	# User, movie info
	user_info_embedding = Flatten()(Embedding(num_user, user_info_len, weights=[user_info], trainable=False)(user))
	movie_info_embedding = Flatten()(Embedding(num_movie, movie_info_len, weights=[movie_info], trainable=False)(movie))
	
	# Concate to get final input weight
	feature = Concatenate()([user_weight, movie_weight, user_info_embedding, movie_info_embedding])
	#feature = Concatenate()([user_embedding, movie_embedding])

	fc1 = Dense(128, activation="relu")(feature)
	fc2 = Dense(64, activation="relu")(fc1)
	fc2_drop = Dropout(dropout)(fc2)
	fc3 = Dense(32, activation="relu")(fc2_drop)
	fc3_drop = Dropout(dropout)(fc3)
	fc4 = Dense(10, activation="relu")(fc3_drop)
	fc4_drop = Dropout(dropout)(fc4)

	if classification is False:
		pred = Dense(1)(fc4_drop)
		loss = 'mean_squared_error'
	else:
		pred = Dense(5, activation="softmax")(fc4_drop)
		loss = 'categorical_crossentropy'

	model = Model(inputs=[user, movie], outputs=pred)
	model.compile('adam', loss, metrics=["accuracy"])
	model.summary()
	return model


def multitask_model(latent_dim, num_user, num_movie, predicted_info_shape, dropout, classification=False):
	user, movie, user_weight, _, movie_weight, _ = embedding(num_user, num_movie, latent_dim, dropout)
	feature = Concatenate()([user_weight, movie_weight])

	fc1 = Dense(128, activation="relu")(feature)
	fc1_drop = Dropout(dropout)(fc1)
	fc2 = Dense(128, activation="relu")(fc1_drop)
	fc2_drop = Dropout(dropout)(fc2)

	# Predict score
	fc3 = Dense(32, activation="relu")(fc2_drop)
	fc3_drop = Dropout(dropout)(fc3)
	fc4 = Dense(10, activation="relu")(fc3_drop)
	fc4_drop = Dropout(dropout)(fc4)
	pred = Dense(1, activation="softmax", name="score")(fc4_drop)

	# Predict information
	info = Dense(predicted_info_shape, activation="softmax", name="aux")(fc2_drop)

	model = Model(inputs=[user, movie], outputs=[pred, info])
	model.compile('adam',
								loss={'score': 'mean_squared_error', 'aux': 'categorical_crossentropy'},
								loss_weights={'score': 1, 'aux': 0},
								metrics=["accuracy"])

	model.summary()
	return model
