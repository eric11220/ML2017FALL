import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten

# For ResNet50
from keras import layers
from keras.layers import BatchNormalization, Input, ZeroPadding2D, AveragePooling2D, Reshape, Lambda

from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.regularizers import l2


# Default to channels_last
bn_axis = 3
img_rows = 48
img_cols = 48

def identity_block(input_tensor, kernel_size, filters, stage, block):
	filters1, filters2, filters3 = filters


	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
	filters1, filters2, filters3 = filters

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x


def resize_broker(x, height=224, width=224, data_format="channels_last"):
	from keras.backend import resize_images
	hf = 224/img_rows
	wf = 224/img_cols
	return resize_images(x, hf, wf, data_format)


def resnet50():

	input_shape = (224, 224, 1)
	img_input = Input(shape=input_shape)

	#x = Lambda(resize_broker)(img_input)

	x = ZeroPadding2D((3, 3))(img_input)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	x = AveragePooling2D((7, 7), name='avg_pool')(x)

	x = Flatten()(x)
	x = Dense(7, activation='softmax', name='fc7')(x)

	# Create model.
	model = Model(img_input, x, name='resnet50')

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',
				  optimizer=ada,
				  metrics=['accuracy'])
	model.summary()
	input("check model summary")
	return model


def add_conv_block(model, n_kernel, ksize, block_num, conv_num, activation="relu", batch_norm=False, input_shape=None):
	conv_name = 'block' + str(block_num) + "_conv" + str(conv_num)
	batch_name = 'bn_b' + str(block_num) + '_c' + str(conv_num)
	if input_shape is not None:
		model.add(Conv2D(n_kernel, (ksize, ksize), padding='same', input_shape=input_shape, name=conv_name))
	else:
		model.add(Conv2D(n_kernel, (ksize, ksize), padding='same', name=conv_name))

	if activation == "relu":
		model.add(Activation("relu"))
	elif activation == "PReLU":
		model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	elif activation == "LeakyReLU":
		model.add(keras.layers.LeakyReLU())

	if batch_norm is True:
		model.add(BatchNormalization(name=batch_name))

def add_fc_block(model, n_kernel, fc_num, activation="relu"):
	name = "fc" + str(fc_num)
	model.add(Dense(n_kernel, name=name))
	if activation == "relu":
		model.add(Activation("relu"))
	elif activation == "PReLU":
		model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	elif activation == "LeakyReLU":
		model.add(keras.layers.LeakyReLU())

def vgg16(loss="categorical_crossentropy", dropout=0.2, batch_norm=True, relu="LeakyReLU"):
	input_shape = (img_rows, img_cols, 1)

	# Block 1
	model = Sequential()
	add_conv_block(model, 64, 3, 1, 1, activation=relu, batch_norm=batch_norm, input_shape=input_shape)
	add_conv_block(model, 64, 3, 1, 2, activation=relu, batch_norm=batch_norm)
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	# Block 2
	add_conv_block(model, 128, 3, 2, 1, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 128, 3, 2, 2, activation=relu, batch_norm=batch_norm)
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	# Block 3
	add_conv_block(model, 256, 3, 3, 1, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 256, 3, 3, 2, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 256, 3, 3, 3, activation=relu, batch_norm=batch_norm)
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	# Block 4
	add_conv_block(model, 512, 3, 4, 1, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 512, 3, 4, 2, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 512, 3, 4, 3, activation=relu, batch_norm=batch_norm)
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	# Block 5
	add_conv_block(model, 512, 3, 5, 1, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 512, 3, 5, 2, activation=relu, batch_norm=batch_norm)
	add_conv_block(model, 512, 3, 5, 3, activation=relu, batch_norm=batch_norm)
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

	# Classification block
	model.add(Flatten(name='flatten'))

	add_fc_block(model, 4096, 1, activation=relu)
	model.add(Dropout(dropout))

	add_fc_block(model, 4096, 2, activation=relu)
	model.add(Dropout(dropout))

	add_fc_block(model, 1000, 3, activation=relu)
	model.add(Dropout(dropout))

	if loss == 'categorical_crossentropy':
		model.add(Dense(7, activation='softmax', name='predictions'))
	else:
		model.add(Dense(7, activation='sigmoid', name='predictions'))

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss=loss,
				  optimizer=ada,
				  metrics=['accuracy'])
	model.summary()
	return model


def fc_model(loss):
	model = Sequential()
	model.add(Dense(2000, input_shape=(img_rows*img_cols, ), activation='relu'))
	for _ in range(5):
		model.add(Dense(2000, activation='relu'))
	for _ in range(12):
		model.add(Dense(1000, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(7))
	model.add(Activation('softmax'))

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss=loss,
				  optimizer=ada,
				  metrics=['accuracy'])

	model.summary()
	return model


def orig_model():
	model = Sequential()
	model.add(Conv2D(64, (5, 5), padding='same',
							input_shape=(img_rows, img_cols, 1)))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2), padding='same'))
	  
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2), padding='same'))
	 
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	 
	model.add(keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(2, 2), padding='same'))
	 
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(keras.layers.PReLU(alpha_initializer='zeros'))
	model.add(Dropout(0.2))
	 
	model.add(Dense(7))
	model.add(Activation('softmax'))

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',
				  optimizer=ada,
				  metrics=['accuracy'])
	model.summary()
	return model
