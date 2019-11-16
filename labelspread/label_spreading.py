import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature, io, transform
from skimage.io import imread, imshow, imsave
from scipy import ndimage
from skimage.transform import rescale, resize

from sklearn.semi_supervised import LabelSpreading

import random

def load_dataset(data_frame, image_type):
	#HYPERPARAMETER FOR UNCERTAINTY DUE TO 'NO FINDING'
	df = data_frame.fillna(value={'No Finding':0,'Pleural Effusion': 0})

	#GIVEN FRONTAL OR LATERAL IMAGE TYPE ONLY
	if(image_type!='All'):
		df_IMAGE = df.where(df['Frontal/Lateral']==image_type).dropna(subset =['Frontal/Lateral'])
	else:
		df_IMAGE = df

	max_img = 2500
	new_h = int(320)
	new_w = int(370)

	data_x = np.zeros((max_img,new_h,new_w))

	# Read in all images that we will use
	i = 0
	for path in df_IMAGE['Path']:
	    # reading the image
		img = plt.imread("./" + path)/255.
	    # converting the type of pixel to float 32
		img = img.astype('float64')
	    # appending the image into the list
		img = transform.resize(img, (new_h, new_w))
		data_x[i,:,:] = np.array(img)
		i+=1
		if(i%100 == 0):
			print("Processed", i, "of", max_img, "images.")
		if(i>=max_img):
			print(path)
			break

	# defining the target
	data_y = df_IMAGE['Pleural Effusion'][:max_img].values

	train_x = data_x[:int(i*0.7),:,:]
	val_x = data_x[int(i*0.7)+1:i,:,:]
	train_y = data_y[:int(i*0.7)]
	val_y = data_y[int(i*0.7)+1:i]

	train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
	val_x = val_x.reshape(val_x.shape[0], val_x.shape[1] * val_x.shape[2])
	return train_x, train_y, val_x, val_y

print("Getting data")
data_frame = pd.read_csv('filtered_train.csv')
train_x, train_y, val_x, val_y = load_dataset(data_frame, "Frontal")
print("Data processed")
model = LabelSpreading(kernel="knn", n_jobs=-1)
model.fit(train_x, train_y)
test_y = model.predict(val_x)
out_df = pd.DataFrame({"Expected": val_y, "Output": test_y})
out_df.to_csv("label_spreading_results.csv")
print("Process complete")