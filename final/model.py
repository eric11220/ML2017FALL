from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, dot, Embedding

def build_model(seqlen, wordvec):
	num_wordvec, veclen = wordvec.shape

	embedding = Embedding(num_wordvec, veclen, weights=[wordvec], trainable=False, mask_zero=True)
	lstm = LSTM(256, return_sequences=False)

	first = Input(shape=(seqlen,))
	first_embed = embedding(first)
	first_encoded = lstm(first_embed)

	second = Input(shape=(seqlen,))
	second_embed = embedding(second)
	second_encoded = lstm(second_embed)

	dotted = dot([first_encoded, second_encoded], 1)

	model = Model(inputs=[first, second], outputs=dotted)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])
	model.summary()
	return model
