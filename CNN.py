import tensorflow as tf






def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')


def max_pool(x, kernel_size=[1,2,2,1], strides=[1,2,2,1]):
	return tf.nn.max_pool(x,ksize=kernel_size, strides=strides, padding='SAME')


class CNN():
	def __init__(self, x, class_num=120, keep_prob=1.0):
		self.class_num = class_num
		self.keep_prob = keep_prob
		self.model = self.cnn(x)

	def cnn(self, x):
		#conv1
		w_conv1 = weight_variable([5,5,3,32])
		b_conv1 = bias_variable([32])
		# x_image = tf.reshape(x, [-1, img_w, img_h, 3])
		h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
		h_pool1 = max_pool(h_conv1, kernel_size=[1,3,3,1])

		#conv2
		w_conv2 = weight_variable([5,5,32,64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
		h_pool2 = max_pool(h_conv2, kernel_size=[1,3,3,1], strides=[1,1,1,1])

		#full connect
		w_fc1 = weight_variable([32*32*64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
		# h_pool2_flat = tf.layers.flatten(h_pool2)
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

		#dropout
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

		#output
		w_fc2 = weight_variable([1024, self.class_num])
		b_fc2 = bias_variable([self.class_num])
		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

		return y_conv


