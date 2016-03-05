import os
import cv2
import numpy as np

FACE_DIR = "faces2"
DESIRED_MEAN = 127.5
DESIRED_STD = 40.0


def normalize(img):
	img_copy = img.copy()
	mean = img_copy.mean()
	std = img_copy.std()
	for x in np.nditer(img_copy, op_flags=['readwrite']):
		pixel = (x - mean) * DESIRED_STD/std + DESIRED_MEAN
		if pixel > 255:
			pixel = 255
		x[...] = pixel
	return img_copy

N = 0

#Create Empty List for normalized images
imgArray = []

# Iterate through files in faceDir
for subdir, dirs, files in os.walk(FACE_DIR):
	for file in files:
		filePath = os.path.join(subdir, file)
		img = cv2.imread(filePath, 0)
		#Add normalized image to list
		imgArray.append(normalize(img))
		N += 1

print N

