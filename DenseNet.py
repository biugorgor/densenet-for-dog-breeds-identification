import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import pandas as pd




def conv_layer(input,filter,kernel,stride=1, layer_name='conv'):
	with tf.name_scope(layer_name):
		network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
		return network


def Global_Average_Pooling(x, stride=1):
	# width = np.shape(x)[1]
	# height = np.shape(x)[2]
	# pool_size = [width, height]
	# return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

	return global_avg_pool(x, name='Global_avg_pooling')


def BatchNormalization(x, training, scope):
	with arg_scope([batch_norm],
					scope=scope,
					updates_collections=None,
					decay=0.9,
					center=True,
					scale=True,
					zero_debias_moving_mean=True):
		return tf.cond(training,
						lambda:batch_norm(inputs=x, is_training=training, reuse=None),
						lambda:batch_norm(inputs=x, is_training=training, reuse=True))


def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
	return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)



class DenseNet():
	def __init__(self, x, nb_blocks, dropout_rate, class_num, filters, training):
		self.nb_blocks = nb_blocks
		self.dropout_rate = dropout_rate
		self.class_num = class_num
		self.filters = filters
		self.training = training
		self.model = self.dense_net(x)

	def bottleneck_layer(self, x, scope):
		with tf.name_scope(scope):
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = tf.nn.relu(x)
			x = conv_layer(x, filter=4*self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)

			x = BatchNormalization(x, training=self.training, scope=scope+'_batch2')
			x = tf.nn.relu(x)
			x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
			x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)

			return x

	def transition_layer(self, x, scope):
		with tf.name_scope(scope):
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = tf.nn.relu(x)
			x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
			x = Average_pooling(x, pool_size=[2,2], stride=2)

			return x

	def dense_block(self, input_x, nb_layers, layer_name):
		with tf.name_scope(layer_name):
			layers_concat = list()
			layers_concat.append(input_x)

			x = self.bottleneck_layer(input_x, scope=layer_name+'_bottleN_'+str(0))
			layers_concat.append(x)

			for i in range(nb_layers-1):
				x = tf.concat(layers_concat, axis=3)
				x = self.bottleneck_layer(x, scope=layer_name+'_bottleN_'+str(i+1))
				layers_concat.append(x)
			x = tf.concat(layers_concat,axis=3)
			return x

	def dense_net(self, input_x):
		x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		x = Max_Pooling(x, pool_size=[3,3], stride=2)

		for i in range(self.nb_blocks):
			x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
			x = self.transition_layer(x, scope='trans_'+str(i))

		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

		x = BatchNormalization(x, training=self.training, scope='linear_batch')
		x = tf.nn.relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = tf.layers.dense(inputs=x, units=self.class_num, name='linear')

		return x





