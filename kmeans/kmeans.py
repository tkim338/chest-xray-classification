import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans

from skimage import feature

from skimage.io import imread, imshow, imsave
from scipy import ndimage
from skimage.transform import rescale, resize

from sklearn.decomposition import PCA

import random

#from sklearn.datasets import load_digits
#digits = load_digits()
#print('data shape: '+str(digits.data.shape))

num_clusters = 20
img_dim = 240
scale = 2

def count_num_members(labels):
	unique, counts = np.unique(labels, return_counts=True)
	num_labels = len(unique)
	output = np.zeros([num_labels,1])
	for i in range(num_labels):
		output[i] = counts[i]
	return output

def blob_extraction(img, sigma=1, T=50, scale=2):
	filtered_img = ndimage.gaussian_filter(img, sigma)
	labeled, nr_objects = ndimage.label(filtered_img > T)
	resized_img = resize(labeled, (labeled.shape[0] // scale, labeled.shape[1] // scale), anti_aliasing=False)
	return resized_img

data_length = (img_dim//scale)**2
data = np.zeros([1,data_length])
patient_list_file = open('./CheXpert-v1.0-small/filtered_train.csv')
patient_list_file.readline()
count = 0
for patient_row in patient_list_file:
	# frontal only
	if (patient_row.split(',')[3] == "Frontal"):

		rand = random.random()
		if (rand < 0.01): # 1% data = 1,000 rows = 200MB
			if (rand < 0.0001):
				print(rand)
			filename = './'+patient_row.split(',')[0]
			#print(filename)
			img = imread(filename)
			border = [(img.shape[0] - img_dim)//2, (img.shape[1] - img_dim)//2]
			img = img[border[0]:img_dim+border[0],border[1]:img_dim+border[1]] # crop image, take center
			#edges = img
			edges = blob_extraction(img, scale)
			data_row = edges.flatten()
			
			data_row.resize([1,data_length])

			data = np.vstack([data,data_row]) # truncate to uniform size (75x80)
		#count+=1
		#if (count > 20):
		#	break
print("data compiled")

from sys import getsizeof
print(getsizeof(data))

data = data[1:,:]
print(data.shape)

pca = PCA(0.99, whiten=True)
data = pca.fit_transform(data)
print(data.shape)
new_dim = data.shape[1]

#kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=100, max_iter=10)

clusters = kmeans.fit_predict(data)
print("clusters assigned")
print('kmeans cluster centers shape: '+str(kmeans.cluster_centers_.shape))

'''
f_output = open("kmeans_centers_"+str(num_clusters)+"_clusters.csv","w")
for c in range(kmeans.cluster_centers_.shape[0]):
	outputStr = str(c)
	for f in range(kmeans.cluster_centers_.shape[1]):
		outputStr += ','+str(kmeans.cluster_centers_[c,f])
	outputStr += "\n"
	f_output.write(outputStr)
f_output.close()
'''

'''
fig, ax = plt.subplots(2, int(round(num_clusters/2)), figsize=(8, 3))
#centers = kmeans.cluster_centers_.reshape(num_clusters, img_dim//scale, img_dim//scale) # 8, 8)
centers = kmeans.cluster_centers_.reshape(num_clusters, new_dim, new_dim) # 8, 8)

for axi, center, num_members in zip(ax.flat, centers, range(num_clusters)):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest')#, cmap=plt.cm.binary)
    axi.title.set_text('num members: '+str(int(num_members)))

fig.tight_layout()

plt.show()
'''
patient_list_file = open('./CheXpert-v1.0-small/filtered_train.csv')
header = patient_list_file.readline() # skips header

f_output = open("kmeans_predictions_"+str(num_clusters)+"_clusters.csv","w")
f_output.write(header[0:len(header)-1]+',Cluster Assignment\n')

count = 0
for patient_row in patient_list_file:
	predictionStr = patient_row[0:len(patient_row)-1]
	
	filename = './'+patient_row.split(',')[0]
	img = imread(filename)
	border = [(img.shape[0] - img_dim)//2, (img.shape[1] - img_dim)//2]
	img = img[border[0]:img_dim+border[0],border[1]:img_dim+border[1]] # crop image, take center
	edges = blob_extraction(img, scale)
	data_row = edges.flatten()
	data_row.resize([1,data_length])
	
	data_row = pca.transform(data_row)
	
	pred = kmeans.predict(data_row)
	predictionStr += ','+str(int(pred))+'\n'
	f_output.write(predictionStr)
	
	#count+=1
	#if (count > 200):
	#	break
		
f_output.close()
print("prediction done")
