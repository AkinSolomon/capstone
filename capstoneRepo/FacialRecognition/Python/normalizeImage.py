# Script for creating the training set for facial recognition using PCA
# Note: All images in FACE_DIR must be the same size
# Principle Author: Teddy Reehorst

import os
import cv2
import numpy as np

#FACE_DIR is the folder that holds all of the training set
FACE_DIR = "faces1"
#Desired Mean is the goal of the mean after normalization
DESIRED_MEAN = 127.5
#Desired STD is the goal STD after normalization
DESIRED_STD = 60.0

#Normalizes a copy of the img arguement
def normalize(img):
	img_copy = img.copy()
	#Calculate mean and STD of img
	mean = img_copy.mean()
	std = img_copy.std()
	#Iterate through the pixels in the image
	for x in np.nditer(img_copy, op_flags=['readwrite']):
		pixel = (x - mean) * DESIRED_STD/std + DESIRED_MEAN
		#Limit the Pixel value to a maximum of 255
		if pixel > 255:
			pixel = 255
		x[...] = pixel
	return img_copy


#Create Empty List for normalized images
imgArray = []

# Iterate through files in faceDir
for subdir, dirs, files in os.walk(FACE_DIR):
	# Iterate through the files in FACE_DIR
	for file in files:
		filePath = os.path.join(subdir, file)
		img = cv2.imread(filePath, 0)
		#Add normalized image to list
		imgArray.append(normalize(img))



#Calculate Average img
N = len(imgArray)
avgImg = np.zeros_like(imgArray[0])
tot = 0
for img in imgArray:
	avgImg += img/N
	tot += img.mean()
print avgImg.shape
print avgImg.mean()
print tot/N

# Create an array to hold difference images
diffArray = np.zeros_like(imgArray)
n = 0
for img in imgArray:
	diffArray[n,:,:] = img - avgImg
	cv2.imshow('diff' + str(n), img - avgImg)
	n += 1


cv2.imshow('average',avgImg)
cv2.waitKey(0)
cv2.destroyAllWindows()




	

