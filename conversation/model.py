from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Embedding, Lambda

'''
	seq_shape: (max_len, veclen) = (13, 100)
	num_words: Number of word choices
'''
def seq2seq(seqlen, num_words, wordvec, onehot, latent_dim=512, batch=64):
	num_wordvec, veclen = wordvec.shape
	embedding = Embedding(num_wordvec, veclen, weights=[wordvec], trainable=False, mask_zero=True)

	onehot_embedding = Embedding(num_words, num_words, weights=[onehot], trainable=False, mask_zero=True)

	# Encoding
	e_input = Input(batch_shape=(batch, seqlen))
	encoder_input = embedding(e_input)
	encoder = LSTM(latent_dim, return_state=True, stateful=True)
	encoded, state_h, state_c = encoder(encoder_input)
	encoder_state = [state_c, state_h]

	# Decoding
	d_input = Input(batch_shape=(batch, seqlen+1))
	decoder_input = onehot_embedding(d_input)
	decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
	decoded, _, _ = decoder(decoder_input, initial_state=encoder_state)
	output = Dense(num_words, activation='softmax')(decoded)
	output = Lambda(lambda x: x[:, :-1, :])(output)
	
	model = Model([e_input, d_input], output)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
	return model
