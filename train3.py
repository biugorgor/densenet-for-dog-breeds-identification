import functools
import tensorflow as tf
from src import DenseNet3




slim = tf.contrib.slim



networks_map = {
	'densenet121': DenseNet3.densenet121,
	'densenet161': DenseNet3.densenet161,
	'densenet169': DenseNet3.densenet169,
}

arg_scope_map = {
	'densenet121': DenseNet3.densenet_arg_scope,
	'densenet161': DenseNet3.densenet_arg_scope,
	'densenet169': DenseNet3.densenet_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, data_format='NHWC',is_training=False):
	
	if name not in networks_map:
		raise ValueError('Name of network unknown %s' % name)
	
	arg_scope = arg_scope_map[name](weight_decay=weight_decay, data_format=data_format)
	func = networks_map[name]
	
	@functools.wraps(func)
	def network_fn(images):
		with slim.arg_scope(arg_scope):
			return func(images, num_classes,data_format=data_format, is_training=is_training)

	if hasattr(func, 'default_image_size'):
		network_fn.default_image_size = func.default_image_size

	return network_fn




