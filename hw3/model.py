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

def vgg16(loss="categorical_crossentropy", dropout=0.2):
	input_shape = (img_rows, img_cols, 1)

	# Block 1
	model = Sequential()
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='block1_conv1'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	# Block 2
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	# Block 3
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	# Block 4
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	# Block 5
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

	# Classification block
	model.add(Flatten(name='flatten'))
	model.add(Dense(4096, activation='relu', name='fc1'))
	model.add(Dropout(dropout))
	model.add(Dense(4096, activation='relu', name='fc2'))
	model.add(Dropout(dropout))
	model.add(Dense(1000, activation='relu', name='fc3'))
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
