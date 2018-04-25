import tensorflow as tf
import numpy as np



class DNN():
	def __init__(self, x, img_size, batch_size=64, class_num=120, layers_depth=5):
		self.class_num = class_num
		self.layers_depth = layers_depth
		self.img_size = img_size
		self.batch_size = batch_size
		self.model = self.dnn(x)

	def dnn(self, x):
		W = []
		B = []
		a = x

		weight = tf.Variable(np.random.randn(4096, self.img_size) / tf.sqrt(self.img_size * 1.0), dtype=tf.float32,name="W" + str(1))
		W.append(weight)
		bias = tf.Variable(np.zeros((4096, self.batch_size)), dtype=tf.float32, name="b" + str(1))
		B.append(bias)
		y = tf.matmul(weight, a) + bias
		a = tf.nn.relu(y)

		weight = tf.Variable(np.random.randn(2048, 4096) / tf.sqrt(2048 * 1.0), dtype=tf.float32,name="W" + str(1))
		W.append(weight)
		bias = tf.Variable(np.zeros((2048, self.batch_size)), dtype=tf.float32, name="b" + str(1))
		B.append(bias)
		y = tf.matmul(weight, a) + bias
		a = tf.nn.relu(y)

		weight = tf.Variable(np.random.randn(1024, 2048) / tf.sqrt(2048 * 1.0), dtype=tf.float32,name="W" + str(1))
		W.append(weight)
		bias = tf.Variable(np.zeros((1024, self.batch_size)), dtype=tf.float32, name="b" + str(1))
		B.append(bias)
		y = tf.matmul(weight, a) + bias
		a = tf.nn.relu(y)

		with tf.name_scope('hidden_layers'):
			for i in range(2,self.layers_depth-1):
				weight = tf.Variable(np.random.randn(1024, 1024) / tf.sqrt(1024 * 1.0), dtype=tf.float32,name="W" + str(i))
				W.append(weight)
				bias = tf.Variable(np.zeros((1024, self.batch_size)), dtype=tf.float32, name="b" + str(i))
				B.append(bias)
				y = tf.matmul(weight, a) + bias
				a = tf.nn.relu(y)

		weight = tf.Variable(np.random.randn(self.class_num, 1024) / tf.sqrt(1024 * 1.0), dtype=tf.float32,name="W" + str(self.layers_depth))
		bias = tf.Variable(np.zeros((self.class_num, self.batch_size)), dtype=tf.float32, name="b" + str(self.layers_depth))
		y = tf.matmul(weight, a) + bias

		return y


