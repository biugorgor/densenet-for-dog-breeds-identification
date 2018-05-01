import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time


curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data')

model_dir = os.path.join(curr_dir,'model')

# ckpt_path = os.path.join(model_dir,'dense121\\dense_121_300\\dense121.ckpt')
ckpt_path = os.path.join(model_dir,'resnet\\ResNeXt.ckpt')

pb_model_path = os.path.join(model_dir, 'dense_121.pb')

class_num = 120
img_width = 64
img_height = 64




def one_hot_decode(one_hot_label):
	train_y = pd.read_csv(data_dir+'/breeds.csv',dtype={'breed':np.str})
	lb = preprocessing.LabelBinarizer()
	lb.fit(train_y['breed'])
	return np.asarray(lb.inverse_transform(one_hot_label), dtype=np.str)


def get_img(img_path):
	img_raw = open(img_path,'rb').read()
	img = tf.cast(img_raw, dtype=tf.string)
	img = tf.image.decode_jpeg(img_raw, channels=3)
	img = tf.image.resize_image_with_crop_or_pad(img,img_width,img_height)
	# img = tf.image.resize_image_with_crop_or_pad(img,256,256)
	img = tf.image.per_image_standardization(img)
	img = tf.reshape(img,[-1,img_width,img_height,3])
	img = tf.cast(img, tf.float32)*(1./255)-0.5

	return img


def classify(img_path):
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(ckpt_path+".meta")
		saver.restore(sess, ckpt_path)
		x = sess.graph.get_tensor_by_name('x:0')
		y = sess.graph.get_tensor_by_name('Softmax:0')
		y_ = sess.graph.get_tensor_by_name('ArgMax:0')
		# drop_out = sess.graph.get_tensor_by_name('dropout_rate:0')
		# training = sess.graph.get_tensor_by_name('training_flag:0')
		training = sess.graph.get_tensor_by_name('Placeholder_1:0')

		img = get_img(img_path)

		# probs, predicts = sess.run([y,y_], feed_dict={drop_out:0.0,
		# 								x:sess.run(img),
		# 								training:False})
		
		probs, predicts = sess.run([y,y_], feed_dict={x:sess.run(img),training:False})
		
		breeds = one_hot_decode(np.identity(class_num)).reshape(-1)
		df = pd.DataFrame(data={'prob':probs.reshape(-1), 'breed':breeds})
		return df.sort_values(['prob'],ascending=False)
		

def load_model(pb_model_path):
	if not os.path.exists(pb_model_path):
		raise ValueError(pb_model_path + " is not exist.")
	with tf.gfile.FastGFile(pb_model_path, 'rb') as fid:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fid.read())
		for node in graph_def.node:
			if node.op == 'RefSwitch':
				node.op = 'Switch'
				for index in range(len(node.input)):
					if 'moving_' in node.input[index]:
						node.input[index] = node.input[index] + '/read'
			elif node.op == 'AssignSub':
				node.op = 'Sub'
				if 'use_locking' in node.attr: del node.attr['use_locking']

		tf.import_graph_def(graph_def, name='')
		model_graph = tf.get_default_graph()
		return model_graph


def classify2(img_path):
	model_graph = load_model(pb_model_path)
	with model_graph.as_default():
		with tf.Session(graph=model_graph) as sess:
			inputs = model_graph.get_tensor_by_name('x:0')
			x = sess.graph.get_tensor_by_name('x:0')
			y = sess.graph.get_tensor_by_name('Softmax:0')
			drop_out = sess.graph.get_tensor_by_name('dropout_rate:0')
			training = sess.graph.get_tensor_by_name('Placeholder_1:0')

			img = get_img(img_path)

			probs = sess.run([y], feed_dict={drop_out:0.0,
											x:sess.run(img),
											training:True})
			
			breeds = one_hot_decode(np.identity(class_num)).reshape(-1)
			df = pd.DataFrame(data={'prob':probs.reshape(-1), 'breed':breeds})
			return df.sort_values(['prob'],ascending=False)



if __name__ == '__main__':
	# img_dir = 'images'
	# for name in os.listdir(img_dir):
	# 	start_time = time.time()
	# 	img_path = os.path.join(img_dir, nmae)
	# 	result = classify(img_path).take(list(range(5)))
	# 	end_time = time.time()
	# 	print(result)
	# 	print('totally cost:', end_time - start_time)

	start_time = time.time()
	img_path = 'images/dog2.jpeg'
	result = classify(img_path).take(list(range(5)))
	end_time = time.time()
	print(result)
	print('totally cost:', end_time - start_time)


