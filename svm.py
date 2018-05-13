from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer
from sklearn.externals import joblib
# import cPickle as pickle
import numpy as np
import cv2 as cv



def saveModel(model, path):
	joblib.dump(model, path)


def loadModel(path):
	return joblib.load(path)


if __name__ == '__main__':
	model_file = 'model/svm/svm_model.pkl'
	x = [[1,2],[2,4],[4,5],[3,2],[3,1]]
	x1 = [[2,2],[5,4],[3,7]]
	y = [0,0,1,1,2]
	clf = OneVsRestClassifier(estimator=SVC(random_state=0, gamma=0.5))
	model = clf.fit(x,y)
	saveModel(model, model_file)
	model1 = loadModel(model_file)
	pred = model1.predict(x)
	print(pred)
	pred = model1.predict(x1)
	print(pred)

	# iris = datasets.load_iris()
	# x, y = iris.data, iris.target
	# clf = OneVsRestClassifier(LinearSVC(random_state=0))
	# pred = clf.fit(x,y).predict(x)
	# print(pred)

