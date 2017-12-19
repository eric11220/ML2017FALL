from keras.models import Model, Sequential
from keras.layers import dot, add, Concatenate 
from keras.layers import  Input, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding

def embedding(num_user, num_movie, latent_dim, dropout):
	user = Input(shape=[1])
	movie = Input(shape=[1])

	user_weight = Flatten()(Embedding(num_user, latent_dim)(user))
	#user_weight = Dropout(dropout)(user_weight)

	movie_weight = Flatten()(Embedding(num_movie, latent_dim)(movie))
	#movie_weight = Dropout(dropout)(movie_weight)

	movie_bias = Flatten()(Embedding(num_movie, 1)(movie))
	user_bias = Flatten()(Embedding(num_user, 1)(user))
	return user, movie, user_weight, user_bias, movie_weight, movie_bias


def mf(num_user, num_movie, latent_dim, dropout):
	user, movie, user_weight, user_bias, movie_weight, movie_bias = embedding(num_user, num_movie, latent_dim, dropout)

	dot_result = dot([user_weight, movie_weight], axes=-1)
	added_user_bias = add([dot_result, user_bias])
	added_movie_bias = add([added_user_bias, movie_bias])

	model = Model(inputs=[user, movie], outputs=added_movie_bias)
	model.compile('adam', 'mean_squared_error', metrics=["accuracy"])
	model.summary()
	return model


def nn(latent_dim, dropout, user_info, movie_info, classification=False):
	num_user, user_info_len = user_info.shape
	num_movie, movie_info_len = movie_info.shape

	# Latent representation
	user, movie, user_weight, _, movie_weight, _ = embedding(num_user, num_movie, latent_dim, dropout)

	# User, movie info
	user_info_embedding = Flatten()(Embedding(num_user, user_info_len, weights=[user_info], trainable=False)(user))
	movie_info_embedding = Flatten()(Embedding(num_movie, movie_info_len, weights=[movie_info], trainable=False)(movie))

	user_embedding = Dense(latent_dim, activation="relu")(user_info_embedding)
	movie_embedding = Dense(latent_dim, activation="relu")(movie_info_embedding)
	
	# Concate to get final input weight
	feature = Concatenate()([user_weight, movie_weight, user_info_embedding, movie_info_embedding])
	#feature = Concatenate()([user_embedding, movie_embedding])

	fc1 = Dense(128, activation="relu")(feature)
	fc1_drop = Dropout(dropout)(fc1)
	fc2 = Dense(64, activation="relu")(fc1_drop)
	fc2_drop = Dropout(dropout)(fc2)
	fc3 = Dense(10, activation="relu")(fc2_drop)
	fc3_drop = Dropout(dropout)(fc3)

	if classification is False:
		pred = Dense(1, activation="relu")(fc3_drop)
		loss = 'mean_squared_error'
	else:
		pred = Dense(5, activation="softmax")(fc3_drop)
		loss = 'categorical_crossentropy'

	model = Model(inputs=[user, movie], outputs=pred)
	model.compile('adam', loss, metrics=["accuracy"])
	model.summary()
	return model
