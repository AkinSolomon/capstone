import numpy as np
import cv2

img = cv2.imread("messi5.jpg",0)

org = img.copy()

img = img.astype('float')


mean = float(img.mean())
std = float(img.std())

meanDes = 127.5
stdDes = 40.0

print mean
print std

img = (img - meanDes)*std/stdDes + mean

print img.shape

print meanDes
print img.mean()

print img.std()
print stdDes

cv2.imshow('original',org)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
