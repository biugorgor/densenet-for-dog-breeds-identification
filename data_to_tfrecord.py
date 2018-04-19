import os
import xml.etree.ElementTree
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing


curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data')
# data_train_dir = os.path.join(data_dir,'train')
data_train_dir = os.path.join(data_dir,'train_resize')
data_test_dir = os.path.join(data_dir,'test')
# tf_train_data = os.path.join(data_dir,'train.tfrecords')
tf_train_data = os.path.join(data_dir,'train_resize.tfrecords')
tf_test_data = os.path.join(data_dir,'test.tfrecords')



def int64_feature(value):
	return tf.train.Feature(bytes_list=tf.train.Int64List(value=[value]))

def float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_label():
	train_y = pd.read_csv(os.path.join(data_dir,'labels.csv'),dtype={'id':np.str, 'breed':np.str})
	lb = preprocessing.LabelBinarizer()
	lb.fit(train_y['breed'])
	# print(lb.classes_)
	# print(lb.transform(train_y['breed']).reshape(-1))
	# print(label)

	def encode(labels):
		return np.asarray(lb.transform(labels),dtype=np.float32)

	return train_y,encode



def make_tf_record():
	with tf.Graph().as_default(),tf.Session() as sess:
		# for data in [tf_train_data,tf_test_data]:

		writer = tf.python_io.TFRecordWriter(tf_train_data)

		y,encoder = get_label()
		for i in range(0,len(y['id'])):
			img_file = y['id'][i]
			path = os.path.join(data_train_dir,img_file+'.jpg')
			image = open(path,'rb').read()

			breed = y['breed'][i]

			def one_label():
				return {
					'breed':breed,
					'id':img_file
				}

			one_label = one_label()
			one_hot_label = encoder([one_label['breed']]).reshape(-1).tolist()

			example = tf.train.Example(features=tf.train.Features(feature={
				'one_hot_label':float_feature(one_hot_label),
				'label':bytes_feature(breed.encode()),
				'image_raw':bytes_feature(image)
				}))

			writer.write(example.SerializeToString())
			print(img_file,breed,'done')

		writer.flush()
		writer.close()
		print('finished!')


if __name__ == '__main__':
	# np.set_printoptions(threshold=np.nan)
	make_tf_record()
