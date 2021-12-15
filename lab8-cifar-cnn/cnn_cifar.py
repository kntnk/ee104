
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore") # Hide the warning at execution

#print('tensorflow version:',tf.__version__) # Show the version of Tensorflow

# Create the convolutional base
def model_layers():
	model = models.Sequential()
	# 3x Convolution with 64 different filters in size of (3x3)
	model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(layers.Dropout(rate=0.2))
	# 3x Convolution with 128 different filters in size of (3x3)
	model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(layers.Dropout(rate=0.3))
	# 3x Convolution with 256 different filters in size of (3x3)
	model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(layers.Dropout(rate=0.4))
	# 3x Convolution with 512 different filters in size of (3x3)
	model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(layers.Dropout(rate=0.5))
	# Add Dense layers on top
	model.add(layers.Flatten())
	model.add(layers.Dense(units=10, activation='softmax'))
	model.summary()
	return model



# Apply an arbitrary learning rate depending on the number of epochs
def step_decay(epoch):
	lr = 0.001 # learning rate
	if (epoch >= 75): lr /= 10.0 # lr = 0.0001
	return lr



def cnn_demo():
	# Download and prepare the CIFAR10 dataset
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

	"""
	print(train_images.shape)	# (50000, 32, 32, 3)
	print(train_labels.shape)	# (50000, 1)
	print(test_images.shape)	# (10000, 32, 32, 3)
	print(test_labels.shape)	# (10000, 1)
	"""

	# Normalize the image in the range 0-1
	train_images = train_images.astype('float32') / 255.0
	test_images  = test_images.astype('float32') / 255.0

	# One-hot encoding
	train_labels = to_categorical(train_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	model = model_layers()

	bs = 128 # = batch_size

	# Compile and train the model
	model.compile(
		optimizer=Adam(),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	# data augmentation
	datagen = ImageDataGenerator(
		rotation_range=20,		# Degree range for random rotations
		width_shift_range=0.2,
		height_shift_range=0.2,
		#zoom_range=0.1,		# Range for random zoom
		channel_shift_range=0.2,# Range for random channel shifts
		horizontal_flip=True,	# Randomly flip inputs horizontally.
		#vertical_flip=True,	# Randomly flip inputs vertically
	).flow(train_images, train_labels, batch_size=bs)

	valgen = ImageDataGenerator().flow(test_images, test_labels, batch_size=bs)

	lr_decay = LearningRateScheduler(step_decay)

	history = model.fit_generator(
		generator=datagen,
		epochs=100,
		steps_per_epoch=train_images.shape[0] // bs,
		validation_data=valgen,
		validation_steps=test_images.shape[0] // bs,
		callbacks=[lr_decay],
	)
	
	# Evaluate the model
	fig, axs = plt.subplots(1, 2)
	axs[0].plot(history.history['accuracy'], label='accuracy - training')
	axs[0].plot(history.history['val_accuracy'], label = 'val_accuracy - validation')
	axs[0].set_xlabel('Epoch')
	axs[0].set_ylabel('Accuracy')
	axs[0].legend()

	axs[1].plot(history.history['loss'], label='loss - training')
	axs[1].plot(history.history['val_loss'], label = 'val_loss - validation')
	axs[1].set_xlabel('Epoch')
	axs[1].set_ylabel('Loss')
	axs[1].legend()
	plt.show()

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print(test_acc * 100, "%")


if __name__ == "__main__":
	cnn_demo()
