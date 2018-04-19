#coding=utf8
# _*_coding:utf-8_*_

from PIL import Image
import os


images_root_dir = 'data/train'
images2_root_dir = 'data/train_resize'


def resize(filename):
	path = os.path.join(images_root_dir, filename)
	path2 = os.path.join(images2_root_dir,  filename)
	if os.path.exists(path2):
		os.remove(path2)
	img = Image.open(path)
	img = img.convert('RGB')
	resized = img.resize((128,128),Image.BILINEAR)
	resized.save(path2)


if __name__ == '__main__':
	if not os.path.exists(images2_root_dir):
		os.mkdir(images2_root_dir)
	for filename in os.listdir(images_root_dir):
		print(filename)
		image = resize(filename)
	print('Finished')
