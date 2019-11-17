import numpy as np
import pandas as pd
from scipy import sparse
from skimage import io, transform
from functools import reduce
#Improvements
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#Improvements
import torch
from torch.utils.data import TensorDataset, Dataset


def load_dataset(data_frame, image_type, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	#HYPERPARAMETER FOR UNCERTAINTY DUE TO 'NO FINDING'
	df = data_frame.fillna(value={'No Finding':0,'Pleural Effusion': 0})

	#GIVEN FRONTAL OR LATERAL IMAGE TYPE ONLY
	if(image_type!='All'):
		df_IMAGE = df.where(df['Frontal/Lateral']==image_type).dropna(subset =['Frontal/Lateral'])
	else:
		df_IMAGE = df

	max_img = 25000
	new_h = int(320)
	new_w = int(370)

	data_x = np.zeros((max_img,new_h,new_w))

	i = 0
	for path in df_IMAGE['Path']:
	    # reading the image
		img = plt.imread("../" + path)/255.
	    # converting the type of pixel to float 32
		img = img.astype('float64')
	    # appending the image into the list
		img = transform.resize(img, (new_h, new_w))
		data_x[i,:,:] = np.array(img)
		i+=1
		if(i>=max_img):
			print(path)
			break
	# defining the target
	data_y = df_IMAGE['Pleural Effusion'].values
	data_y = data_y[:max_img]

	train_x = data_x[:int(i*0.7),:,:]
	val_x = data_x[int(i*0.7)+1:i,:,:]
	train_y = data_y[:int(i*0.7)]
	val_y = data_y[int(i*0.7)+1:i]

	# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.3)

	size_test = 5000
	test_x = np.zeros((size_test,new_h,new_w))
	j = 0
	print('1 D')
	for pathtest in df_IMAGE['Path'][max_img+1:]:
		# reading the image
		img = plt.imread("../" + pathtest)/255.
	    # converting the type of pixel to float 32
		img = img.astype('float64')
	    # appending the image into the list
		img = transform.resize(img, (new_h, new_w))
		test_x[j,:,:] = np.array(img)
		j+=1
		if(j>=size_test):
			print(pathtest)
			break
	print(test_x.shape)
	test_y = df_IMAGE['Pleural Effusion'][max_img+1:max_img+j+1].values
	print(test_y.shape)
	data_train = torch.from_numpy(train_x).unsqueeze(1)
	target_train = torch.from_numpy(train_y.astype('long'))
	dataset_train = TensorDataset(data_train, target_train)

	data_val = torch.from_numpy(val_x).unsqueeze(1)
	target_val = torch.from_numpy(val_y.astype('long'))
	dataset_val = TensorDataset(data_val, target_val)

	data_test = torch.from_numpy(test_x).unsqueeze(1)
	target_test = torch.from_numpy(test_y.astype('long'))
	dataset_test = TensorDataset(data_test, target_test)

	print('DONE')

	return dataset_train, dataset_val, dataset_test
