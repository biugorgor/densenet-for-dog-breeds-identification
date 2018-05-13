import numpy as np
import pandas as pd
import os
import cv2 as cv 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer
from skimage import feature
from sklearn.externals import joblib




curr_dir = os.path.abspath('.')
data_dir = os.path.join(curr_dir,'data/stanford_data')
train_dir = os.path.join(data_dir,'train')
test_dir = os.path.join(data_dir,'test')
train_label_path =  os.path.join(data_dir,'train_label.txt')
test_label_path = os.path.join(data_dir,'test_label.txt')




def saveModel(model, path):
	joblib.dump(model, path)


def loadModel(path):
	return joblib.load(path)


def LBP(img_path, radius, no_points):
	im = cv.imread(img_path)
	im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	lbp = feature.local_binary_pattern(im_gray, no_points, radius, method='default')
	# print(lbp)
	return lbp


def hist(img_path, radius, no_points):
	eps = 1e-7
	im = cv.imread(img_path)
	im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	lbp = feature.local_binary_pattern(im_gray, no_points, radius, method='default')
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, no_points+3), range=(0, no_points+2))
	# print(lbp)
	return hist


def get_LBP(data_dir):
	radius = 3
	no_points = radius*8
	lbp = []
	i=0
	for path in os.listdir(data_dir):
		# if i>=1000:
		# 	break
		# i += 1

		img_path = os.path.join(data_dir,path)
		# lbp = LBP(img_path, radius, no_points)
		lbp1 = hist(img_path, radius, no_points)
		lbp.append(lbp1)

	return lbp


def get_label(path):
	data = pd.read_csv(path)
	label = data['label']

	return np.array(label.tolist())



def main():
	model_path = 'model/svm/svm_model.pkl'

	print('loading train label...')
	train_label = get_label(train_label_path)
	print('loading test label...')
	test_label = get_label(test_label_path)

	print('loading train image, and computing train hist...')
	train_hist = get_LBP(train_dir)
	print('loading test image, and computing test hist...')
	test_hist = get_LBP(test_dir)

	# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	# result = OneVsRestClassifier(svr_rbf,-1).fit(train_hist, train_label).score(test_hist,test_label)
	clf = OneVsRestClassifier(SVC(random_state=0, gamma=0.1), -1)
	print('training model...')
	model = clf.fit(train_hist, train_label)

	print('testing model...')
	result = model.score(test_hist, test_label)
	print('testing acc',result)

	print('saving model...')
	saveModel(model, model_path)
	print('model save in ',model_path,' sucessfully!')



if __name__ == '__main__':
	main()


