from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, Embedding, merge, dot, GRU, Flatten

def build_model(seqlen, wordvec):
    num_wordvec, veclen = wordvec.shape

    embedding = Embedding(num_wordvec, veclen, weights=[wordvec], trainable=True, mask_zero=True)
    lstm_1 = GRU(128, return_sequences=False)
    lstm_2 = GRU(128, return_sequences=False)

    first = Input(shape=(seqlen,))
    first_embed = embedding(first)
    first_encoded = lstm_1(first_embed)

    second = Input(shape=(seqlen,))
    second_embed = embedding(second)
    second_encoded = lstm_2(second_embed)

    dotted = dot([first_encoded, second_encoded], 1) #mode='cos')
    dotted = Activation('sigmoid')(dotted)

    model = Model(inputs=[first, second], outputs=dotted)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])
    model.summary()
    return model
