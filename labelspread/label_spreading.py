import pandas as pd
from skimage import feature
from skimage.io import imread, imshow, imsave
from scipy import ndimage
from skimage.transform import rescale, resize

from sklearn.semi_supervised import LabelSpreading

import random

def getTrainData():
	print("Starting import")
	in_file = pd.DataFrame.read_csv('./filtered_train.csv')
	print("Finished import")
	print(in_file[0])

print("Begin")
getTrainData()