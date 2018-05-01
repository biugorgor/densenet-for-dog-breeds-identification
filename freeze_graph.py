import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os



curr_dir = os.path.abspath('.')
model_dir = os.path.join(curr_dir,'model')

ckpt_path = os.path.join(model_dir,'dense.ckpt')




def main():
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(ckpt_path+".meta")
		saver.restore(sess, ckpt_path)
		graph_def = tf.get_default_graph().as_graph_def()
		
		# for node in graph_def.node:
		# 	if node.op == 'RefSwitch':
		# 		node.op = 'Switch'
		# 		for index in range(len(node.input)):
		# 			if 'moving_' in node.input[index]:
		# 				node.input[index] = node.input[index] + '/read'
		# 	elif node.op == 'AssignSub':
		# 		node.op = 'Sub'
		# 		if 'use_locking' in node.attr: del node.attr['use_locking']

		out_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['Softmax'])

		with tf.gfile.GFile(model_dir+'/dense_121.pb', 'wb') as writer:
			serialized_graph = out_graph_def.SerializeToString()
			writer.write(serialized_graph)
		
	print("done")


if __name__ == '__main__':
	main()