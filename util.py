from __future__ import print_function
# In[1]:
import pickle
import numpy as np
import time
import sys  
sys.path.append('./models')
import matplotlib.pyplot as plt

import keras
from keras import backend as K
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras
import numpy as np
from keras.models import load_model
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
from heapq import nsmallest
from keras.models import Sequential,Model
import time


def normalize(X_train,X_test):
	#this function normalize inputs for zero mean and unit variance
	# it is used when training a model.
	# Input: training set and test set
	# Output: normalized training set and test set according to the trianing set statistics.
	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train, axis=(0, 1, 2, 3))
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)
	return X_train, X_test


def load_data():
	t0 = time.time()  

	(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32
	X_train = X_train.astype('float32')  # uint8-->float32
	X_test = X_test.astype('float32')
	X_train,X_test = normalize(X_train,X_test)
	print('训练样例:', X_train.shape, Y_train.shape,
	      ', 测试样例:', X_test.shape, Y_test.shape)

	nb_classes = 10  # label为0~9共10个类别
	# Convert class vectors to binary class matrices
	Y_train = to_categorical(Y_train, nb_classes)
	Y_test = to_categorical(Y_test, nb_classes)
	print("取数据耗时: %.2f seconds ..." % (time.time() - t0))

	# define generators for training and validation data
    #data augmentation
	train_datagen = ImageDataGenerator(
	    featurewise_center=False,  # set input mean to 0 over the dataset
	    samplewise_center=False,  # set each sample mean to 0
	    featurewise_std_normalization=False,  # divide inputs by std of the dataset
	    samplewise_std_normalization=False,  # divide each input by its std
	    zca_whitening=False,  # apply ZCA whitening
	    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
	    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	    horizontal_flip=True,  # randomly flip images
	    vertical_flip=False)  # randomly flip images

	val_datagen = ImageDataGenerator(
	    featurewise_center=True,
	    featurewise_std_normalization=True)

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	train_datagen.fit(X_train)
	val_datagen.fit(X_test)
	return X_train,Y_train,X_test,Y_test,train_datagen,val_datagen



def train(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen):
	learning_rate = 0.001
	lr_decay = 1e-6
	lr_drop = 20
	def lr_scheduler(epoch):
		return learning_rate * (0.5 ** (epoch // lr_drop))
	reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

	sgd = keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
	model.compile(
	    optimizer=sgd, 
	    loss='categorical_crossentropy', metrics=['accuracy']
	)

	callbacks = [
	    EarlyStopping(monitor='val_acc', patience=7, min_delta=0.01,verbose=1),
	    reduce_lr
	            ]

	batch_size = 32
	model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
	                steps_per_epoch=len(X_train)//batch_size, epochs=10,
	                validation_data=val_datagen.flow(X_test, Y_test, batch_size=batch_size), 
	                validation_steps=len(X_test)//batch_size,
	                callbacks=callbacks, initial_epoch=0, shuffle=True, verbose=2)
	return model



def test(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen):
	model.compile(
	    optimizer=keras.optimizers.Adam(lr=1e-4), 
	    loss='categorical_crossentropy', metrics=['accuracy']
	)
	score = model.evaluate_generator(val_datagen.flow(X_test, Y_test), steps=len(X_test)/256, use_multiprocessing=False, verbose=2)
	loss = score[0]
	acc = score[1]
	print('loss:',loss)
	print('acc:',acc)
	return model,loss,acc
