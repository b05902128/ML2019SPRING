import os
import sys
import numpy as np 
from skimage.io import imread, imsave
def process(M): 
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	return M



IMAGE_DIR_PATH = sys.argv[1]
imagepath = sys.argv[2]
outputpath = sys.argv[3]
# IMAGE_PATH = 'Aberdeen'
# Images for compression & reconstruction
test_image = ['50.jpg','100.jpg','150.jpg','200.jpg','250.jpg'] 
# Number of principal components used
k = 5
filelist = os.listdir(IMAGE_DIR_PATH) 
# Record the shape of images
img_shape = imread(os.path.join(IMAGE_DIR_PATH,filelist[0])).shape
img_data = []
for filename in filelist:
	tmp = imread(os.path.join(IMAGE_DIR_PATH,filename))  
	img_data.append(tmp.flatten())
training_data = np.array(img_data).astype('float32')
print(training_data.shape)
# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 
training_data = training_data.transpose()

# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False) 





#1.c
# for x in test_image: 
# 	# Load image & Normalize
# 	picked_img = imread(os.path.join(IMAGE_PATH,x))  
# 	imsave(x,picked_img)
# 	X = picked_img.flatten().astype('float32') 
# 	X -= mean	
# 	# Compression
# 	weight = np.dot(X,u[:,:5])
# 	# Reconstruction
# 	reconstruct = process(np.dot(weight,u[:,:5].transpose()) + mean)
# 	imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape)) 
# for x in test_image: 
	# Load image & Normalize
picked_img = imread(os.path.join(IMAGE_DIR_PATH,imagepath))  
X = picked_img.flatten().astype('float32') 
X -= mean	
# Compression
weight = np.dot(X,u[:,:5])
# Reconstruction
reconstruct = process(np.dot(weight,u[:,:5].transpose()) + mean)
imsave(outputpath, reconstruct.reshape(img_shape)) 

#1.a
average = process(mean)
imsave('average.jpg', average.reshape(img_shape)) 

#1.b
for x in range(5):
	a = u[:,x]
	eigenface = process(u[:,x])
	imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))

#1.d
for i in range(5):
	number = s[i] * 100 / sum(s)
	print(number)    
