# coding: utf-8

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

from util import normalize,load_data,train,test



''''''''''''''''''''''
name:			computer_prune_config(model)
function：		得到模型卷积核信息(对象,名称，通道数)
parameters:		model -- 模型
return：			model_prune_layers_config -- 卷积核信息
'''''''''''''''''''''
def computer_prune_config(model):
	model_config = model.get_config()
	# print(model_config)
	model_layers_number = len(model_config['layers'])

	model_prune_layers_config = {}
	model_layer = []
	model_layers_true_name = []
	model_layers_true_index = []
	model_layers_true_channle = []
	for index in range(model_layers_number):
		
		model_perlayers_classname = model_config['layers'][index]['class_name']
		model_perlayers_config = model_config['layers'][index]['config']

		if((model_perlayers_classname == 'Conv2D')):
			layer = model.get_layer(index=index)
			model_layer.append(layer)
			model_layers_true_name.append(model_perlayers_config['name'])
			model_layers_true_index.append(index)
			model_layers_true_channle.append(model_perlayers_config['filters'])

	model_prune_layers_config['layer'] = model_layer
	model_prune_layers_config['name'] = model_layers_true_name
	model_prune_layers_config['index'] = model_layers_true_index
	model_prune_layers_config['channle'] = model_layers_true_channle

	return model_prune_layers_config



''''''''''''''''''''''
name:			count_conv2d_filters(layer_config)
function：		计算模型的总卷积核个数
parameters:		layer_config -- 模型中的卷积层信息
return：			number_of_filters -- 总卷积核个数
'''''''''''''''''''''
def count_conv2d_filters(layer_config):
	model_layers_channle = layer_config['channle']
	number_of_filters = 0

	for i in range(len(model_layers_channle)):
		number_of_filters = number_of_filters +  model_layers_channle[i]
	return number_of_filters




''''''''''''''''''''''
name:			computer_activition(model,Conv2D_layer_config,X_test)
function：		计算前向传播激活函数输出并得到相关信息
parameters:		model -- 被剪枝的模型
				Conv2D_layer_config -- 模型中的卷积层信息
				X_test -- 进行前向传播的测试数据
return：			pre_prune_target -- 预剪枝信息(卷积核层名称，卷积核层对象，卷积核通道数)
'''''''''''''''''''''
def computer_activition(model,Conv2D_layer_config,X_test):
	model_config = model.get_config()
	# print(model_config)
	model_layers_number = len(model_config['layers'])
	model_prune_layers_config = {}
	model_layer = []
	model_layers_true_name = []
	model_layers_true_index = []

	for index in range(model_layers_number):		
		model_perlayers_classname = model_config['layers'][index]['class_name']
		model_perlayers_config = model_config['layers'][index]['config']
		if( (model_perlayers_classname == 'Activation') ):
			if((index!=model_layers_number-1) and (index!=model_layers_number-5)):		# 根据自己模型层情况更改，这里我使用的VGG只需要conv2d(1-13)
				layer = model.get_layer(index=index)
				model_layer.append(layer)
				model_layers_true_name.append(model_perlayers_config['name'])
				model_layers_true_index.append(index)

	model_prune_layers_config['layer'] = model_layer
	model_prune_layers_config['name']  = model_layers_true_name
	model_prune_layers_config['index'] = model_layers_true_index

	# 计算前向传播中卷积核经过激活层激活的值
	start = time.time()

	full_model = model
	predictions = []

	for i in range(len(model_prune_layers_config['name'])):
		model = Model(inputs=model.input, outputs=full_model.get_layer(model_prune_layers_config['name'][i]).output)
		prediction = model.predict(X_test)
		predictions.append(prediction)
		# print(prediction.shape)
	print('前向传播耗时：%4d seconds' % (time.time()-start))


	pre_prune_target = {}

	# 计算前向传播中激活函数输出的L2范数
	start = time.time()

	for i in range(len(predictions)):
		# 每层activation的相关参数：1. layer_channle_index：层数  2. layer_weights：预测值(激活值)  3. shape：形状
		layer_channle_index = predictions[i].shape[-1]		# 因为是channle_last
		layer_weights = predictions[i]
		shape = (predictions[i].shape[1], predictions[i].shape[2], predictions[i].shape[-1])

		# 对每层activation的predictions[i].shape[0]个预测值进行均值处理
		zero_weights = np.zeros(shape=(1, predictions[i].shape[1], predictions[i].shape[2], predictions[i].shape[-1]))
		for idx in range(predictions[i].shape[0]):
			zero_weights = zero_weights + np.array(layer_weights[idx,:,:,:])
		zero_weights = zero_weights.reshape(shape)
		normalization_layer_weights = zero_weights / predictions[i].shape[0]

		# 计算每个每层activation的channle的L2范数
		for index in range(layer_channle_index):
			layer_weight = np.array(normalization_layer_weights[:,:,index])
			L2 = layer_weight / np.linalg.norm(layer_weight,ord=2)
			L2 = np.sum(np.reshape(L2,(L2.size,)))
			# L2_result.append(L2)
			pre_prune_target[str(Conv2D_layer_config['name'][i])+'.'+str(index)] = L2    # 建立L2范数结果和卷积层对应关系
		# print('计算还剩 %2d 次'%(len(predictions)-(i+1)))

	# print(L2_result)
	print('激活函数计算耗时：%4d seconds' % (time.time()-start))

	return pre_prune_target



''''''''''''''''''''''
name:			get_real_prune_config(pre_prune_target,prune_channles,Conv2D_layer_config)
function：		得到最低排序后真正要剪枝的卷积核相关信息
parameters:		num_filters_to_prune_per_iteration -- 一次剪枝的卷积核个数
				pre_prune_target -- 被剪枝的目标
				Conv2D_layer_config -- 模型中的卷积层信息
return：			prune_target -- 真正的剪枝信息(卷积核层名称，卷积核层对象，卷积核通道数)
'''''''''''''''''''''
def get_real_prune_config(num_filters_to_prune_per_iteration,pre_prune_target,Conv2D_layer_config):
	# 得到排名最低的512个卷积核通道
	prune_channles = nsmallest(num_filters_to_prune_per_iteration, list(pre_prune_target.values()))

	# 获取键值对信息
	def get_key (dict, value):
		return [k for k, v in dict.items() if v == value]

	# 得到真正要剪枝的卷积核信息
	prune_target = []
	conv2d_name = Conv2D_layer_config['name']
	conv2d_layer = Conv2D_layer_config['layer']
	for i in range(len(prune_channles)):
		prune_conv2d = get_key(pre_prune_target,prune_channles[i])
		if(len(prune_conv2d)==1):
			prune_conv2d_name = prune_conv2d[0].split('.')[0]
			prune_conv2d_layer = conv2d_layer[conv2d_name.index(prune_conv2d[0].split('.')[0])]
			prune_channle = prune_conv2d[0].split('.')[1]
			prune_target.append([prune_conv2d_name, prune_conv2d_layer, prune_channle])

	return prune_target



''''''''''''''''''''''
name:			get_list_number(List)
function：		通过剪枝的卷积核列表得到要写入文件的信息
parameters:		prune_information_path -- 文件路径
				prune_target -- 被剪枝的目标
return：			information -- 写入文件的信息
'''''''''''''''''''''
def get_list_number(List):
	myset = set(List)
	information = {}
	Layer_number = []
	Number_of_pruned_filters_pruned = []
	for item in myset:
		Layer_number.append('conv2d_'+str(item))
		Number_of_pruned_filters_pruned.append(List.count(item))
	information['layer_name'] = Layer_number
	information['layer_number_of_filter'] = Number_of_pruned_filters_pruned

	return information



''''''''''''''''''''''
name:			write_prune_information(iter,prune_information_path,prune_target)
function：		将剪枝的卷积核相关信息写入文件
parameters:		prune_information_path -- 文件路径
				prune_target -- 被剪枝的目标
return：			None
'''''''''''''''''''''
def write_prune_information(iter,prune_information_path,prune_target):
	prune_config = {}
	prune_target_filter_number = []
	with open(prune_information_path,'a') as f:
		info = '***************剪枝迭代次数iter:'+str(iter+1)+'***************'+'\r\n'
		f.write(info)
		f.write('---------------本次剪枝迭代中要剪枝的卷积核信息:\r\n')
		for i in range(len(prune_target)):
			prune_target_filter_number.append(prune_target[i][0].split('_')[1])
		information = get_list_number(prune_target_filter_number)
		f.write('卷积层名称    剪枝的卷积核\r\n')
		for i in range(len(information['layer_name'])):	
			f.write(information['layer_name'][i]+'       '+str(information['layer_number_of_filter'][i])+'\r\n')




if __name__ == '__main__':
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	#config.gpu_options.visible_device_list = "0"
	set_session(tf.Session(config=config))


	#[1]: 	加载已模型
	json_file = open('logs/weights/cifar10-VGG.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = keras.models.model_from_json(loaded_model_json)
	model.load_weights('logs/weights/cifar10-VGG.h5')
	print(model.summary())


	#[2]: 	加载数据
	X_train,Y_train,X_test,Y_test,train_datagen,val_datagen = load_data()


	#[3]:   准备剪枝
	prune = True
	if prune:

		num_filters_to_prune_per_iteration = 512
		prune_information_path = 'logs/prune_information/prune.txt'

		# 得到准备剪枝的 '卷积层' 信息
		Conv2D_layer_config = computer_prune_config(model)

		# 得到总的准备剪枝的 '卷积核' 个数
		number_of_filters = count_conv2d_filters(Conv2D_layer_config)

		# 总共要迭代的次数
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
		iterations = int(iterations * 2.0 / 3)
		print("总共要迭代的次数：", iterations)

		for iter in range(iterations):

			info = '\r\n'+'***************剪枝迭代次数iter:'+str(iter+1)+'***************'
			print(info)
			# 得到准备剪枝的 '卷积层' 信息
			Conv2D_layer_config = computer_prune_config(model)
			print('本次迭代总卷积核信息为：',Conv2D_layer_config)

			# 得到总的准备剪枝的 '卷积核' 个数
			number_of_filters = count_conv2d_filters(Conv2D_layer_config)
			print('本次迭代总卷积核个数为：',number_of_filters)

			# 得到准备剪枝的 '卷积核' 和 '通道' 信息(只计算了各通道L2范数)
			print('进行前向传播...........%')
			pre_prune_target = computer_activition(model=model,Conv2D_layer_config=Conv2D_layer_config,X_test=X_test)

			# 得到真正要剪枝的 '卷积核' 和 '通道' 信息(进行了排序)
			print('进行排序...........%')
			prune_target = get_real_prune_config(num_filters_to_prune_per_iteration,pre_prune_target,Conv2D_layer_config)
			print('本次迭代总共要剪枝的卷积核个数为：',len(prune_target))
			print('本次迭代将要剪枝的卷积核信息存入：',prune_information_path)

			# 将最终确定的剪枝信息写入文件
			print('准备将剪枝信息写入文件...........%')
			write_prune_information(iter,prune_information_path,prune_target)			# print('准备剪枝............%\r\n')


			print('正在剪枝...........%')
			final_prune_target = []
			for i in range(len(Conv2D_layer_config['name'])):
				channle = []
				for index in range(len(prune_target)):
					if(Conv2D_layer_config['name'][i] == prune_target[index][0]):
						if prune_target[index][2]:		# 哪一层的卷积层对应的准备剪枝的卷积核数目不为0才添加
							channle.append(int(prune_target[index][2]))
				if(len(channle)):
					final_prune_target.append([Conv2D_layer_config['name'][i],Conv2D_layer_config['layer'][i],channle])
			# print(final_prune_target)

			for i in range(len(final_prune_target)):
				if(i==0):
					model = delete_channels(model,final_prune_target[i][1],final_prune_target[i][2])
					New_Conv2D_layer_config = computer_prune_config(model)
				else:
					New_index = New_Conv2D_layer_config['name'].index(final_prune_target[i][0])
					model = delete_channels(model,New_Conv2D_layer_config['layer'][New_index],final_prune_target[i][2])
					New_Conv2D_layer_config = computer_prune_config(model)
			print(model.summary())
			# model.save('logs/weights/'+'iter_'+str(iter+1)+'_prune_model.h5')


			print('重新训练............%')
			model = train(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen)
			model, loss, acc = test(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen)
			if(int(acc)>0.9):
				model.save_weights('logs/weights/'+'iter_'+str(iter+1)+'_prune_trained_model.h5')
				model_json = model.to_json()
				with open('logs/weights/iter_'+str(iter+1)+'_prune_trained_model_json.json', "w") as json_file:
					json_file.write(model_json)
				# model.save('logs/weights/'+'iter_'+str(iter+1)+'_prune_trained_model.h5')
				print('剪枝模型已保存！\r\n')
			else:
				model = train(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen)
				model, loss, acc = test(model,X_train,Y_train,X_test,Y_test,train_datagen,val_datagen)
				if(int(acc)>0.9):
					model.save_weights('logs/weights/'+'iter_'+str(iter+1)+'_prune_trained_model.h5')
					model_json = model.to_json()
					with open('logs/weights/iter_'+str(iter+1)+'_prune_trained_model_json.json', "w") as json_file:
						json_file.write(model_json)
					# model.save('logs/weights/'+'iter_'+str(iter+1)+'_prune_trained_model.h5')
					print('剪枝模型已保存！\r\n')
