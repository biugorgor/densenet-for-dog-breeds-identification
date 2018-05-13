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
from src.DenseNet2 import DenseNet





curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data')
data_train_dir = os.path.join(data_dir,'train')
# data_train_dir = os.path.join(data_dir,'train_resize')
data_test_dir = os.path.join(data_dir,'test')
# tf_train_data = os.path.join(data_dir,'stanford.tfrecords')
# tf_train_data = os.path.join(data_dir,'stanford_crop_64.tfrecords')
# tf_test_data = os.path.join(data_dir,'test.tfrecords')
tf_train_data = os.path.join(data_dir,'stanford_train.tfrecords')
tf_test_data = os.path.join(data_dir,'stanford_test.tfrecords')


# Hyperparameter
growth_k = 32
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 0.001
epsilon = 1e-8 # AdamOptimizer epsilon
# dropout_rate = 0.2
batch_size = 32
train_size = 19380
test_size = 1200
class_num =120

# res parameter
depth = 32
res_blocks = 2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

total_epochs = 500


img_width = 128
img_height = 128



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


def read_test_tfrecord(record):
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



def get_batch(sess, tfrecords_path, batch_size=64):
	# print(tfrecords_path)
	filenames = tf.placeholder(tf.string)
	data = tf.data.TFRecordDataset(filenames, compression_type='').map(read_tfrecord)
	data = data.shuffle(buffer_size=10000)
	train_iter = data.repeat().batch(batch_size).make_initializable_iterator()

	sess.run(train_iter.initializer,feed_dict={filenames: tfrecords_path})

	return train_iter.get_next()


def get_test_batch(sess, tfrecords_path, batch_size=64):
	filenames = tf.placeholder(tf.string)
	data = tf.data.TFRecordDataset(filenames, compression_type='').map(read_test_tfrecord)
	test_iter = data.repeat().batch(batch_size).make_initializable_iterator()
	
	sess.run(test_iter.initializer,feed_dict={filenames: tfrecords_path})

	return test_iter.get_next()



def train_with_densenet():

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_width,img_height,3], name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=[None, class_num])

	training_flag = tf.placeholder(tf.bool, name='training_flag')


	learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

	# logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
	logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, class_num=class_num, training=training_flag, dropout_rate=dropout_rate).model
	# logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, class_num=class_num, training=training_flag, dropout_rate=dropout_rate).model_169
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	prediction = tf.nn.softmax(logits, axis=1)

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# tf.summary.scalar('train_loss', cost)
	# tf.summary.scalar('train_accuracy', accuracy)

	saver = tf.train.Saver()

	train_summaries = []
	test_summaries = []

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./model/dense121_001_500_05')
		if ckpt and ckpt.model_ckeckpoint_path:
			saver.restore(sess, ckpt.model_ckeckpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		# merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./logs/dense121_001_500_05_logs', sess.graph)

		epoch_learning_rate = init_learning_rate

		# batch_count = 0
		# for record in tf.python_io.tf_record_iterator(tf_train_data):
		# 	batch_count = batch_count + 1
		# batch_count = int((batch_count - test_size)/batch_size)

		# test_batch_count = int((test_size+batch_size-1) / batch_size)
		batch_count = int(train_size/batch_size)
		test_batch_count = int(test_size/batch_size)

		train_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
		test_batch_next = get_test_batch(sess=sess, tfrecords_path=tf_test_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			# if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
			# 	epoch_learning_rate = epoch_learning_rate / 10
			if epoch > 50 and epoch % 50 == 0:
				if epoch % 100:
					epoch_learning_rate = epoch_learning_rate / 2
				else:
					epoch_learning_rate = epoch_learning_rate / 5

			# train
			train_accuracy = 0.0
			train_loss = 0.0

			for step in range(batch_count):
				batch_features = sess.run(train_batch_next)
				batch_x = batch_features['image_raw']
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']

				train_feed_dict = {
					x: batch_x,
					label: batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : True,
					dropout_rate: 0.5
				}

				_, batch_loss, batch_acc = sess.run([train, cost, accuracy], feed_dict=train_feed_dict)
				train_accuracy += batch_acc
				train_loss += batch_loss 
				if step % 100 == 0:
					print("step:", step, "Loss:", batch_loss, "Accuracy:", batch_acc)

			train_loss /= batch_count
			train_accuracy /= batch_count

			train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
											  tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy)])
			# train_summaries.append(tf.summary.scalar('train_loss', train_loss))
			# train_summaries.append(tf.summary.scalar('train_accuracy', train_accuracy))
			# train_summary = tf.summary.merge(train_summaries)
			line = "epoch:%d, Training loss: %.4f, Training accuracy:%.4f" % (epoch, train_loss,train_accuracy)
			print(line)


			# test
			test_accuracy = 0.0
			test_loss = 0.0


			for step in range(test_batch_count):
				batch_features = sess.run(test_batch_next)
				# batch_features = sess.run(train_batch_next)
				test_batch_x = batch_features['image_raw']
				test_batch_y = batch_features['label_one_hot']
				# test_batch_y = batch_features['one_hot_label']
				test_feed_dict = {
					x: test_batch_x,
					label: test_batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : False,
					dropout_rate: 0.0
				}
				# test_predict = prediction.eval(feed_dict=test_feed_dict)
				test_batch_loss, test_batch_accuracy = sess.run([cost,accuracy], feed_dict=test_feed_dict)
				test_loss += test_batch_loss
				test_accuracy += test_batch_accuracy
				
			test_loss /= test_batch_count
			test_accuracy /= test_batch_count

			test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
											 tf.Summary.Value(tag='test_accuracy', simple_value=test_accuracy)])
			# test_summaries.append(tf.summary.scalar('test_loss', test_loss))
			# test_summaries.append(tf.summary.scalar('test_accuracy', test_accuracy))
			# test_summary = tf.summary.merge(test_summaries)
			test_line = "epoch:%d, Testing loss: %.4f, Testing accuracy:%.4f" % (epoch, test_loss,test_accuracy)
			print(test_line)

			with open('logs/dense121_001_500_05_logs.txt','a') as f:
				f.write(line+'\r\n')
				f.write(test_line+'\r\n')

			writer.add_summary(train_summary, global_step=epoch)
			writer.add_summary(test_summary, global_step=epoch)
			if epoch % 10 == 0 or epoch == total_epochs-1:
				saver.save(sess=sess, save_path='./model/dense121_001_500_05/dense121.ckpt')



def train_with_resnet():

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_width,img_height,3], name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=[None, class_num])

	training_flag = tf.placeholder(tf.bool, name='training_flag')


	learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	# dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

	# logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
	logits = ResNeXt(x=x, training=training_flag, blocks=res_blocks, depth=depth, class_num=class_num).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))


	# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
	# train = optimizer.minimize(cost + l2_loss * weight_decay)
	train = optimizer.minimize(cost)

	prediction = tf.nn.softmax(logits, axis=1)

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	saver = tf.train.Saver(tf.global_variables())


	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./model/resnet')
		if ckpt and tf.train.get_checkpoint_exists(ckpt.model_ckeckpoint_path):
			saver.restore(sess, ckpt.model_ckeckpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		writer = tf.summary.FileWriter('./res_logs', sess.graph)

		epoch_learning_rate = init_learning_rate

		# batch_count = 0
		# for record in tf.python_io.tf_record_iterator(tf_train_data):
		# 	batch_count = batch_count + 1
		# batch_count = int(batch_count/batch_size)

		# test_batch_count = int(10221 / batch_size)+1

		batch_count = int(18000/batch_size)
		test_batch_count = int(2000/batch_size)


		train_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
		# test_batch_next = get_test_batch(sess=sess, tfrecords_path=tf_test_data, batch_size=batch_size)

		for epoch in range(total_epochs):
			if epoch == (total_epochs * 0.25) or epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
			# if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10

			train_accuracy = 0.0
			train_loss = 0.0

			for step in range(batch_count):
				batch_features = sess.run(train_batch_next)
				batch_x = batch_features['image_raw']
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']

				train_feed_dict = {
					x: batch_x,
					label: batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : True,
				}

				_, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
				batch_acc = accuracy.eval(feed_dict=train_feed_dict)

				train_loss += batch_loss
				train_accuracy += batch_acc
				if step % 200 == 0:
					print("step:", step, "Loss:", batch_loss, "Accuracy:", batch_acc)
			
			train_loss /= batch_count
			train_accuracy /= batch_count
			train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
											  tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy)])
			line = "epoch:%d, Training loss: %.4f, Training accuracy:%.4f" % (epoch, train_loss,train_accuracy)
			print(line)

			# test
			test_accuracy = 0.0
			test_loss = 0.0

			for step in range(test_batch_count):
				# batch_features = sess.run(test_batch_next)
				# test_batch_x = batch_features['image_raw']
				# # test_batch_y = batch_features['label_one_hot']
				# test_batch_y = batch_features['one_hot_label']
				batch_features = sess.run(train_batch_next)
				test_batch_x = batch_features['image_raw']
				test_batch_y = batch_features['label_one_hot']

				test_feed_dict = {
					x: test_batch_x,
					label: test_batch_y,
					learning_rate: epoch_learning_rate,
					training_flag : False,
				}
				# test_predict = prediction.eval(feed_dict=test_feed_dict)
				batch_test_loss, batch_test_accuracy = sess.run([cost,accuracy], feed_dict=test_feed_dict)
				test_loss += batch_test_loss
				test_accuracy += batch_test_accuracy
				
			test_loss /= test_batch_count
			test_accuracy /= test_batch_count

			test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
    										 tf.Summary.Value(tag='test_accuracy', simple_value=test_accuracy)])

			test_line = "epoch:%d, Testing loss: %.4f, Testing accuracy:%.4f" % (epoch, test_loss,test_accuracy)
			print(test_line)
			with open('logs/ResNeXt_logs.txt','a') as f:
				f.write(line+'\r\n')
				f.write(test_line+'\r\n')

			writer.add_summary(train_summary, global_step=epoch)
			writer.add_summary(test_summary, global_step=epoch)
			# writer.flush()
			saver.save(sess=sess, save_path='./model/resnet/ResNeXt.ckpt')



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

	saver = tf.train.Saver()

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./model/cnn')
		if ckpt and tf.train.get_checkpoint_exists(ckpt.model_ckeckpoint_path):
			saver.restore(sess, ckpt.model_ckeckpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./cnn_logs', sess.graph)

		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int((batch_count - test_size)/batch_size)

		test_batch_count = int((test_size+batch_size-1) / batch_size)

		train_batch_next, test_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size, test_size=test_size)

		for epoch in range(total_epochs):
			if epoch == epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10

			# train
			train_accuracy = 0.0
			train_loss = 0.0

			for step in range(batch_count):
				batch_features = sess.run(train_batch_next)
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

				_, batch_loss, batch_acc = sess.run([train, cost, accuracy], feed_dict=train_feed_dict)
				train_accuracy += batch_acc
				train_loss += batch_loss 
				if step % 100 == 0:
					print("step:", step, "Loss:", batch_loss, "Accuracy:", batch_acc)

			train_loss /= batch_count
			train_accuracy /= batch_count
			train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
											  tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy)])
			print("epoch:", epoch, "Training loss:", train_loss, "Training accuracy:", train_accuracy)

			# test
			test_accuracy = 0.0
			test_loss = 0.0

			for step in range(test_batch_count):
				batch_features = sess.run(test_batch_next)
				test_batch_x = batch_features['image_raw']
				test_batch_y = batch_features['label_one_hot']
				test_feed_dict = {
					x: test_batch_x,
					label: test_batch_y,
					learning_rate: epoch_learning_rate,
					keep_prob:1.0
				}
				batch_test_loss, batch_test_accuracy = sess.run([cost,accuracy], feed_dict=test_feed_dict)
				test_loss += batch_test_loss
				test_accuracy += batch_test_accuracy
				
			test_loss /= test_batch_count
			test_accuracy /= test_batch_count

			test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
    										 tf.Summary.Value(tag='test_accuracy', simple_value=test_accuracy)])

			print("epoch:", epoch, "Testing loss:", test_loss, "Testing accuracy:", test_accuracy)
			writer.add_summary(train_summary, global_step=epoch)
			writer.add_summary(test_summary, global_step=epoch)

			saver.save(sess=sess, save_path='./model/cnnModel/cnn.ckpt')



def train_with_DNN():

	x = tf.placeholder(dtype=tf.float32, shape=(img_width*img_width*3, None), name='x')
	# batch_images = tf.reshape(x, [-1, img_width, img_width, 3])

	label = tf.placeholder(tf.float32, shape=(class_num, None))

	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	logits = DNN(x=x, img_size=img_width*img_width*3, batch_size=batch_size, class_num=class_num, layers_depth=5).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.transpose(label), logits=tf.transpose(logits)))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	logits = tf.nn.softmax(logits, axis=0)

	correct_prediction = tf.equal(tf.argmax(logits, axis=0), tf.argmax(label, axis=0))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# tf.summary.scalar('loss', cost)
	# tf.summary.scalar('accuracy', accuracy)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./model/dnnModel')
		if ckpt and tf.train.get_checkpoint_exists(ckpt.model_ckeckpoint_path):
			saver.restore(sess, ckpt.model_ckeckpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		# merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./dnn_logs', sess.graph)

		epoch_learning_rate = init_learning_rate

		batch_count = 0
		for record in tf.python_io.tf_record_iterator(tf_train_data):
			batch_count = batch_count + 1
		batch_count = int(batch_count/batch_size + 1)

		test_batch_count = int((test_size+batch_size-1) / batch_size)

		train_batch_next, test_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size, test_size=test_size)

		for epoch in range(total_epochs):
			if epoch == epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10

			# train
			train_accuracy = 0.0
			train_loss = 0.0

			for step in range(batch_count):
				batch_features = sess.run(train_batch_next)
				batch_x = np.transpose(np.reshape(batch_features['image_raw'],(batch_size,-1)))
				# batch_y = batch_features['one_hot_label']
				batch_y = batch_features['label_one_hot']
				batch_label = batch_features['label'].T

				train_feed_dict = {
					x: batch_x,
					label: batch_y.T,
					learning_rate: epoch_learning_rate,
				}

				_, batch_loss, batch_acc = sess.run([train, cost, accuracy], feed_dict=train_feed_dict)
				train_accuracy += batch_acc
				train_loss += batch_loss 
				if step % 100 == 0:
					print("step:", step, "Loss:", batch_loss, "Accuracy:", batch_acc)

			train_loss /= batch_count
			train_accuracy /= batch_count
			train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
											  tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy)])
			print("epoch:", epoch, "Training loss:", train_loss, "Training accuracy:", train_accuracy)

			# test
			test_accuracy = 0.0
			test_loss = 0.0

			for step in range(test_batch_count):
				batch_features = sess.run(test_batch_next)
				test_batch_x = np.transpose(np.reshape(batch_features['image_raw'],(batch_size,-1)))
				test_batch_y = batch_features['label_one_hot']
				test_feed_dict = {
					x: batch_x,
					label: batch_y.T,
					learning_rate: epoch_learning_rate,
				}
				# test_predict = prediction.eval(feed_dict=test_feed_dict)
				batch_test_loss, batch_test_accuracy = sess.run([cost,accuracy], feed_dict=test_feed_dict)
				test_loss += batch_test_loss
				test_accuracy += batch_test_accuracy
				
			test_loss /= test_batch_count
			test_accuracy /= test_batch_count

			test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
    										 tf.Summary.Value(tag='test_accuracy', simple_value=test_accuracy)])


			print("epoch:", epoch, "Testing loss:", test_loss, "Testing accuracy:", test_accuracy)
			writer.add_summary(train_summary, global_step=epoch)
			writer.add_summary(test_summary, global_step=epoch)
			saver.save(sess=sess, save_path='./model/dnnModel/dnn.ckpt')




if __name__ == '__main__':
	train_with_densenet()
	# train_with_resnet()
	# train_with_DNN()
	# train_with_cnn()
