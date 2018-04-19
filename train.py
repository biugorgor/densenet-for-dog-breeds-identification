import os
import tensorflow as tf
import numpy as np
from src.DenseNet import DenseNet




curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data')
# data_train_dir = os.path.join(data_dir,'train')
data_train_dir = os.path.join(data_dir,'train_resize')
data_test_dir = os.path.join(data_dir,'test')
# tf_train_data = os.path.join(data_dir,'train.tfrecords')
tf_train_data = os.path.join(data_dir,'train_resize.tfrecords')
tf_test_data = os.path.join(data_dir,'test.tfrecords')


class_num =120
batch_size = 64
epoch_count = 1000
learning_rate = 0.0001
nb_block = 2
growth_k = 12
epsilon = 1e-8
dropout_rate = 0.2


def get_int64_feature(example,name):
	return int(example.features.feature[name].int64_list.value[0])


def get_float_feature(example, name):
	return int(example.features.feature[name].float_list.value)


def get_bytes_feature(example, name):
	return example.features.feature[name].bytes_list.value[0]


def read_tfrecord(record):
	features = tf.parse_single_example(record, features={
		'one_hot_label': tf.FixedLenFeature([class_num], tf.float32),
		'label': tf.FixedLenFeature([],tf.string),
		'image_raw': tf.FixedLenFeature([],tf.string)
		})
	# img = tf.decode_raw(features['image_raw'], tf.uint8)
	# img = tf.reshape(img, [128, 128, 3])
	img = tf.image.decode_jpeg(features['image_raw'], channels=3)
	img = tf.image.resize_image_with_crop_or_pad(img,128,128)
	img = tf.image.per_image_standardization(img)
	img = tf.cast(img, tf.float32)*(1./255)-0.5
	features['image_raw'] = img

	# img_shape = img.shape
	# print(img_shape)
	# print(img_shape[0])
	# print(img_shape[1])
	# print(img_shape[2])

	# img_shape = features['image_raw'].shape
	# print(img_shape)
	# print(img_shape[0])
	# print(img_shape[1])
	# print(img_shape[2])

	return features


def get_batch(sess, tfrecords_path, batch_size=64, train_sample_size=2000):
	# print(tfrecords_path)
	filenames = tf.placeholder(tf.string)
	data = tf.data.TFRecordDataset(filenames, compression_type='').map(read_tfrecord)
	data_iter = data.repeat().batch(batch_size).make_initializable_iterator()
	# train_sample_iter = data.take(train_sample_size).batch(train_sample_size).make_initializable_iterator()

	sess.run(data_iter.initializer,feed_dict={filenames: tfrecords_path})
	# sess.run(train_sample_iter.initializer,feed_dict={filenames: tfrecords_path})

	# return data_iter.get_next(),train_sample_iter.get_next()
	return data_iter.get_next()



def train():

	training_flag = tf.placeholder(dtype=tf.bool)
	label = tf.placeholder(dtype=tf.float32, shape=[None,class_num], name='label')
	x = tf.placeholder(dtype=tf.float32, shape=[None, 128,128,3], name='x')
	# input_x = tf.reshape(x,[-1,128,128,3])

	logits = DenseNet(x=x, nb_blocks=nb_block, dropout_rate=dropout_rate, class_num=class_num, filters=growth_k,training=training_flag ).model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
	train = optimizer.minimize(cost)

	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)

	saver = tf.train.Saver()


	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state('./model')
		if ckpt and tf.train.get_checkpoint_exists(ckpt.model_ckeckpoint_path):
			saver.restore(sess, ckpt.model_ckeckpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./logs', sess.graph)

		data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
		# data_batch_next, train_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
		# train_sample = sess.run(train_batch_next)
		# train_label = train_sample['label']
		# train_one_hot_label = train_sample['one_hot_label']

		for epoch in range(0, epoch_count):
			batch_features = sess.run(data_batch_next)
			batch_x = batch_features['image_raw']
			batch_y = batch_features['one_hot_label']

			train_feed_dict = {
				x: batch_x,
				label: batch_y,
				training_flag: True
			}

			_, loss = sess.run([train, cost], feed_dict=train_feed_dict)

			if epoch==epoch_count or epoch%10==0:
				saver.save(sess=sess, save_path='./model/dense.ckpt')
				train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)

				print('Step:',epoch, 'Loss:',loss, 'Training accuracy:',train_accuracy)
				writer.add_summary(train_summary,epoch)




if __name__ == '__main__':
	train()

	
	# with tf.Session() as sess:
	# 	# data_batch_next, train_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
	# 	data_batch_next = get_batch(sess=sess, tfrecords_path=tf_train_data, batch_size=batch_size)
	# 	for epoch in range(0, 1):
	# 		batch_features = sess.run(data_batch_next)
	# 		imgs = batch_features['image_raw']
	# 		batch_label = batch_features['label']
	# 		batch_y = batch_features['one_hot_label']
	# 		print(type(imgs[0][0]))
	# 		print(type(batch_label[0][0]))
	# 		print(type(batch_y[0][0]))
	# 		# for batch_feature in batch_features:
	# 			# print(batch_label,batch_y)
	# 			# print(batch_y.tolist())
	

