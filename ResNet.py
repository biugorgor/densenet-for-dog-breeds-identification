import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np






def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
	with tf.name_scope(layer_name):
		network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
		return network


def Global_Average_Pooling(x, stride=1):
	return global_avg_pool(x, name='Global_avg_pooling')


def BatchNormalization(x, training, scope):
	with arg_scope([batch_norm],
				   scope=scope,
				   updates_collections=None,
				   decay=0.9,
				   center=True,
				   scale=True,
				   zero_debias_moving_mean=True) :
		return tf.cond(training,
					   lambda : batch_norm(inputs=x, is_training=training, reuse=None),
					   lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
	return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
	return tf.nn.relu(x)

def Sigmoid(x):
	return tf.nn.sigmoid(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
	return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers):
	return tf.concat(layers, axis=3)

def Fully_connected(x, units=120, layer_name='fully_connected'):
	with tf.name_scope(layer_name) :
		return tf.layers.dense(inputs=x, units=units)




class ResNeXt():

	def __init__(self, x, training, blocks=3, depth=64, class_num=120):
		self.depth = depth
		self.blocks = blocks
		self.training = training
		self.class_num = class_num
		self.model = self.Build_SEnet(x)


	def first_layer(self, x, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=64, kernel=[3,3], stride=1, layer_name=scope+'_conv1')
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)

			return x


	def transform_layer(self, x, stride, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=self.depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)

			x = conv_layer(x, filter=self.depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch2')
			x = Relu(x)

			return x


	def transition_layer(self, x, out_dim, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			# x = Relu(x)
			return x


	def split_layer(self, x, stride, cardinality=4, layer_name='split_layer'):
		with tf.name_scope(layer_name):
			layers_split = list()
			for i in range(cardinality):
				splits = self.transform_layer(x, stride=stride, scope=layer_name+'_splitN_'+str(i))
				layers_split.append(splits)

			return Concatenation(layers_split)


	def squeeze_excitation_layer(self, x, out_dim, ratio, layer_name):
		with tf.name_scope(layer_name):
			squeeze = Global_Average_Pooling(x)
			excitation = Fully_connected(squeeze, units=out_dim/ratio, layer_name=layer_name+'_fc1')
			excitation = Relu(excitation)
			excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fc2')
			excitation = Sigmoid(excitation)
			excitation = tf.reshape(excitation, [-1,1,1,out_dim])
			scale = x * excitation

			return scale


	def residual_layer(self, input_x, out_dim, layer_num, res_block=3, reduction_ratio=4):
		for i in range(res_block):
			input_dim = int(np.shape(input_x)[-1])

			if input_dim * 2 == out_dim:
				flag = True
				stride = 2
				channel = input_dim // 2
			else:
				flag = False
				stride = 1

			x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
			x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
			x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

			if flag is True :
				pad_input_x = Average_pooling(input_x)
				pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
			else :
				pad_input_x = input_x

			input_x = Relu(x + pad_input_x)

		return input_x


	def Build_SEnet(self, input_x):
		input_x = self.first_layer(input_x, scope='first_layer')

		x = self.residual_layer(input_x, out_dim=64, res_block=self.blocks, layer_num='1')
		x = self.residual_layer(x, out_dim=64, res_block=self.blocks, layer_num='2')
		x = self.residual_layer(x, out_dim=64, res_block=self.blocks, layer_num='3')

		x = Global_Average_Pooling(x)
		x = flatten(x)

		x = Fully_connected(x, units=self.class_num, layer_name='final_fully_connected')
		return x

