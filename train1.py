import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import pandas as pd
from sklearn import preprocessing

import os
from src.DNN import DNN
from src.CNN import CNN
from src.ResNet import ResNeXt





curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data')
data_train_dir = os.path.join(data_dir,'train')
# data_train_dir = os.path.join(data_dir,'train_resize')
data_test_dir = os.path.join(data_dir,'test')
# tf_train_data = os.path.join(data_dir,'stanford.tfrecords')
tf_train_data = os.path.join(data_dir,'stanford_crop_128.tfrecords')
tf_test_data = os.path.join(data_dir,'test.tfrecords')


# Hyperparameter
growth_k = 32
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 0.001
epsilon = 1e-8 # AdamOptimizer epsilon
# dropout_rate = 0.2
batch_size = 16
class_num =120

# Momentum Optimizer will use
# nesterov_momentum = 0.9
# weight_decay = 1e-4

total_epochs = 100


img_width = 128
img_height = 128



def one_hot_decode(one_hot_label):
	train_y = pd.read_csv(data_dir+'/breeds.csv',dtype={'breed':np.str})
	lb = preprocessing.LabelBinarizer()
	lb.fit(train_y['breed'])
	return np.asarray(lb.transform(tf.cast(one_hot_label,dtype=np.uint32)),dtype=np.str)


def get_int64_feature(example,name):
	return int(example.features.feature[name].int64_list.value[0])


def get_float_feature(example, name):
	return int(example.features.feature[name].float_list.value)


def get_bytes_feature(example, name):
	return example.features.feature[name].bytes_list.value[0]


def read_tfrecord(record):
	features = tf.parse_single_example(record, features={
		'label_one_hot': tf.FixedLenFeature([class_num], tf.float32),
		# 'one_hot_label': tf.FixedLenFeature([class_num], tf.float32),
		'label': tf.FixedLenFeature([],tf.string),
		'image_raw': tf.FixedLenFeature([],tf.string),
		# 'inception_output':tf.FixedLenFeature([2048], tf.float32)
		})
	# img = tf.decode_raw(features['image_raw'], tf.uint8)
	# img = tf.reshape(img, [128, 128, 3])
	img = tf.image.decode_jpeg(features['image_raw'], channels=3)
	img = tf.image.resize_image_with_crop_or_pad(img,img_width,img_height)
	# img = tf.image.resize_image_with_crop_or_pad(img,256,256)
	img = tf.image.per_image_standardization(img)
	img = tf.cast(img, tf.float32)*(1./255)-0.5
	features['image_raw'] = img

	return features


def get_batch(sess, tfrecords_path, batch_size=64, train_sample_size=2000,test_sample_size=1000):
	# print(tfrecords_path)
	filenames = tf.placeholder(tf.string)
	data = tf.data.TFRecordDataset(filenames, compression_type='').map(read_tfrecord)
	data = data.shuffle(buffer_size=10000)
	data_iter = data.repeat().batch(batch_size).make_initializable_iterator()
	# train_sample_iter = data.take(train_sample_size).batch(train_sample_size).make_initializable_iterator()

	sess.run(data_iter.initializer,feed_dict={filenames: tfrecords_path})
	# sess.run(train_sample_iter.initializer,feed_dict={filenames: tfrecords_path})

	# return data_iter.get_next(),train_sample_iter.get_next()
	return data_iter.get_next()



def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
	with tf.name_scope(layer_name):
		network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
		return network

def Global_Average_Pooling(x, stride=1):
	# width = np.shape(x)[1]
	# height = np.shape(x)[2]
	# pool_size = [width, height]
	# return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
	# It is global average pooling without tflearn

	return global_avg_pool(x, name='Global_avg_pooling')
	# But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
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

def Drop_out(x, rate=0.2, training=True) :
	return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
	return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
	return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
	return tf.concat(layers, axis=3)

def Linear(x) :
	return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
	def __init__(self, x, nb_blocks, filters, training=True, dropout_rate=0.2):
		self.nb_blocks = nb_blocks
		self.filters = filters
		self.training = training
		self.dropout_rate = dropout_rate
		self.model = self.Dense_net(x)


	def bottleneck_layer(self, x, scope):
		# print(x)
		with tf.name_scope(scope):
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
			x = Relu(x)
			x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
			# x = Drop_out(x, rate=self.dropout_rate, training=self.training)
			x = Drop_out(x, training=self.training)

			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
			x = Relu(x)
			x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
			x = Drop_out(x, rate=self.dropout_rate, training=self.training)

			# print(x)

			return x

	def transition_layer(self, x, scope):
		with tf.name_scope(scope):
			x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
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

	def Dense_net(self, input_x):
		x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
		x = Max_Pooling(x, pool_size=[3,3], stride=2)

		# for i in range(self.nb_blocks) :
		#     # 6 -> 12 -> 48
		#     x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
		#     x = self.transition_layer(x, scope='trans_'+str(i))

		x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
		x = self.transition_layer(x, scope='trans_1')
		x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
		x = self.transition_layer(x, scope='trans_2')
		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_3')
		x = self.transition_layer(x, scope='trans_3')
		
		x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

		x = Batch_Normalization(x, training=self.training, scope='linear_batch')
		x = Relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		x = tf.layers.dense(inputs=x, units=1024, name='linear1')
		x = tf.layers.dense(inputs=x, units=class_num, name='linear2')

		return x



def train_with_densenet():

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_width,img_height,3], name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=[None, class_num])

	training_flag = tf.placeholder(tf.bool)


	learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

	# logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
	logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag, dropout_rate=dropout_rate).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	prediction = tf.nn.softmax(logits, axis=1)

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./dlogs', sess.graph)

		global_step = 0
		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int(batch_count/batch_size + 1)

		data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			if epoch == (total_epochs * 0.25) or epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10


			for step in range(batch_count):
				batch_features = sess.run(data_batch_next)
				batch_x = batch_features['image_raw']
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']

				train_feed_dict = {
					x: batch_x,
					label: batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : True,
					dropout_rate: 0.3
				}

				_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				training_flag : True,
				dropout_rate: 0.0
			}
			train_summary = merged.eval(feed_dict=train_feed_dict)
			train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
			predict = prediction.eval(feed_dict=train_feed_dict)
			loss = cost.eval(feed_dict=train_feed_dict)

			for i in range(0,len(predict)):
				# print('label:',['label'][i])
				# print('predict:',one_hot_decode(predict[i]))
				a = tf.argmax(batch_y[i], axis=0)
				b = tf.argmax(predict[i], axis=0)
				print('label,predict:',sess.run([a,b,tf.equal(a,b)]))

			print("epoch:", epoch, "Loss:", loss, "Training accuracy:", train_accuracy)
			writer.add_summary(train_summary, global_step=epoch)



def train_with_resnet():

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_width,img_height,3], name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=[None, class_num])

	training_flag = tf.placeholder(tf.bool)


	learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

	# logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
	logits = ResNeXt(x=x, training=training_flag, class_num=class_num).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))


	l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
	train = optimizer.minimize(cost + l2_loss * weight_decay)

	prediction = tf.nn.softmax(logits, axis=1)

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		graph_writer = tf.summary.FileWriter('./dlogs', sess.graph)

		global_step = 0
		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int(batch_count/batch_size + 1)

		data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			if epoch == (total_epochs * 0.25) or epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10


			for step in range(batch_count):
				batch_features = sess.run(data_batch_next)
				batch_x = batch_features['image_raw']
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']

				train_feed_dict = {
					x: batch_x,
					label: batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : True,
					dropout_rate: 0.3
				}

				_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				training_flag : True,
				dropout_rate: 0.0
			}
			train_summary = merged.eval(feed_dict=train_feed_dict)
			train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
			predict = prediction.eval(feed_dict=train_feed_dict)
			loss = cost.eval(feed_dict=train_feed_dict)

			for i in range(0,len(predict)):
				# print('label:',['label'][i])
				# print('predict:',one_hot_decode(predict[i]))
				a = tf.argmax(batch_y[i], axis=0)
				b = tf.argmax(predict[i], axis=0)
				print('label,predict:',sess.run([a,b,tf.equal(a,b)]))

			print("epoch:", epoch, "Loss:", loss, "Training accuracy:", train_accuracy)
			writer.add_summary(train_summary, global_step=epoch)



def train_with_cnn():
	x = tf.placeholder(dtype=tf.float32, shape=(None, img_width,img_width,3), name='x')
	label = tf.placeholder(tf.float32, shape=(None, class_num))
	keep_prob = tf.placeholder("float")
	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	logits = CNN(x=x, class_num=class_num, keep_prob=keep_prob).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(label, axis=1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./dlogs', sess.graph)

		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int(batch_count/batch_size + 1)

		data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10


			for step in range(batch_count):
				batch_features = sess.run(data_batch_next)
				# batch_x = batch_features['inception_output']
				batch_x = batch_features['image_raw']
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']
				batch_label = batch_features['label']

				train_feed_dict = {
					x: batch_x,
					label: batch_y,
					learning_rate: epoch_learning_rate,
					keep_prob:0.8
				}

				_, loss = sess.run([train, cost], feed_dict=train_feed_dict)
				# train.run(feed_dict=train_feed_dict)

			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				keep_prob:1.0
			}
			train_summary = merged.eval(feed_dict=train_feed_dict)
			loss = cost.eval(feed_dict=train_feed_dict)
			train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
			lgs = logits.eval(feed_dict=train_feed_dict)
			for i in range(0,len(lgs)):
				# print('label:',['label'][i])
				# print('predict:',one_hot_decode(lgs[i]))
				a = tf.argmax(batch_y[i], axis=0)
				b = tf.argmax(lgs[i], axis=0)
				print('label,predict:',sess.run([a,b,tf.equal(a,b)]))

			print("epoch:", epoch, "Loss:", loss, "Training accuracy:", train_accuracy)
			writer.add_summary(train_summary, global_step=epoch)



def train_with_DNN():

	x = tf.placeholder(dtype=tf.float32, shape=(img_width*img_width*3, None), name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=(class_num, None))

	training_flag = tf.placeholder(tf.bool)


	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	# logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
	logits = DNN(x=x, img_size=img_width*img_width*3, batch_size=batch_size, class_num=class_num, layers_depth=3).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.transpose(label), logits=tf.transpose(logits)))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	logits = tf.nn.softmax(logits, axis=0)

	correct_prediction = tf.equal(tf.argmax(logits, axis=0), tf.argmax(label, axis=0))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./dlogs', sess.graph)

		global_step = 0
		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int(batch_count/batch_size + 1)

		data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10


			for step in range(batch_count):
				batch_features = sess.run(data_batch_next)
				# batch_x = batch_features['inception_output']
				batch_x = np.transpose(np.reshape(batch_features['image_raw'],(batch_size,-1)))
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']
				batch_label = batch_features['label'].T

				train_feed_dict = {
					x: batch_x,
					label: batch_y.T,
					learning_rate: epoch_learning_rate,
					training_flag : True
				}

				_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

				if step % 100 == 0:
					global_step += 100
					train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
					# accuracy.eval(feed_dict=feed_dict)
					lgs = tf.transpose(logits).eval(feed_dict=train_feed_dict)
					for i in range(0,len(lgs)):
						# print('label:',['label'][i])
						# print('predict:',one_hot_decode(lgs[i]))
						a = tf.argmax(batch_y[i], axis=0)
						b = tf.argmax(lgs[i], axis=0)
						print('label,predict:',sess.run([a,b,tf.equal(a,b)]))

					print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
					writer.add_summary(train_summary, global_step=epoch)



if __name__ == '__main__':
	train_with_densenet()
	# train_with_resnet()
	# train_with_DNN()
	# train_with_cnn()
