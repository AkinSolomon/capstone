import numpy as np
import cv2

desiredMean = 80.0
desiredSTD = 40.0

img = cv2.imread("messi5.jpg",0)

org = img.copy()

mean = img.mean()
std = img.std()

print mean
print std


for x in np.nditer(img, op_flags=['readwrite']):
	pixel = (x - mean) * desiredSTD/std + desiredMean
	if pixel > 255:
		pixel = 255
	x[...] = pixel

print img.mean()
print img.std()

cv2.imshow('original',org)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
