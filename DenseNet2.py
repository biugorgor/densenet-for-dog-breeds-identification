import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import numpy as np




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

def Drop_out(x, rate=0.0, training=True) :
	return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
	return tf.nn.relu(x)

def Concatenation(layers) :
	return tf.concat(layers, axis=3)



class DenseNet():
	def __init__(self, x, nb_blocks, filters, class_num=120, training=True, dropout_rate=0.2):
		self.nb_blocks = nb_blocks
		self.filters = filters
		self.training = training
		self.class_num = class_num
		self.dropout_rate = dropout_rate
		self.model = self.dense_net_121(x)
		# self.model_169 = self.dense_net_169(x)
		# self.model_201 = self.dense_net_201(x)


	def bottleneck_layer(self, x, scope):
		# print(x)
		with tf.name_scope(scope):
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)
			x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = Drop_out(x, rate=self.dropout_rate, training=self.training)
			# x = Drop_out(x, training=self.training)

			x = BatchNormalization(x, training=self.training, scope=scope+'_batch2')
			x = Relu(x)
			x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
			x = Drop_out(x, rate=self.dropout_rate, training=self.training)

			# print(x)

			return x

	def transition_layer(self, x, scope):
		with tf.name_scope(scope):
			x = BatchNormalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)
			x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			x = Drop_out(x, rate=self.dropout_rate, training=self.training)
			x = Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME')

			return x

	def dense_block(self, input_x, nb_layers, layer_name):
		with tf.name_scope(layer_name):
			layers_concat = list()
			layers_concat.append(input_x)

			x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

			layers_concat.append(x)

			for i in range(nb_layers - 1):
				x = Concatenation(layers_concat)
				x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
				layers_concat.append(x)

			x = Concatenation(layers_concat)

			return x

	def dense_net_121(self, input_x):
		x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		x = Max_Pooling(x, pool_size=[3,3], stride=2)

		x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
		x = self.transition_layer(x, scope='trans_1')
		x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
		x = self.transition_layer(x, scope='trans_2')
		x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
		x = self.transition_layer(x, scope='trans_3')
		
		x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')

		x = BatchNormalization(x, training=self.training, scope='linear_batch')
		x = Relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = tf.layers.dense(inputs=x, units=1024, name='linear1')
		x = tf.layers.dense(inputs=x, units=self.class_num, name='linear2')

		return x
'''
	def dense_net_169(self, input_x):
		x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		x = Max_Pooling(x, pool_size=[3,3], stride=2)

		x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
		x = self.transition_layer(x, scope='trans_1')
		x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
		x = self.transition_layer(x, scope='trans_2')
		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_3')
		x = self.transition_layer(x, scope='trans_3')
		
		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

		x = BatchNormalization(x, training=self.training, scope='linear_batch')
		x = Relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = tf.layers.dense(inputs=x, units=1024, name='linear1')
		x = tf.layers.dense(inputs=x, units=class_num, name='linear2')

		return x

	def dense_net_201(self, input_x):
		x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		# x = Max_Pooling(x, pool_size=[3,3], stride=2)

		x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
		x = self.transition_layer(x, scope='trans_1')

		x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
		x = self.transition_layer(x, scope='trans_2')

		x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
		x = self.transition_layer(x, scope='trans_3')

		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

		x = BatchNormalization(x, training=self.training, scope='linear_batch')
		x = Relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = Drop_out(x, rate=self.dropout_rate, training=self.training)
		x = tf.layers.dense(inputs=x, units=1024, name='linear1')
		x = tf.layers.dense(inputs=x, units=self.class_num, name='linear2')

		return x
'''