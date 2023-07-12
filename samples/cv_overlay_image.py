
import cv2
img1 = cv2.imread('resources/blue.png')
img1 = cv2.resize(img1, (2000, 2000))
img2 = cv2.imread('../human.png')

print(img2.shape)
height = img2.shape[0]
width = img2.shape[1]


# # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
img1[0:height, 0:width,:] = img2 [0:height, 0:width,:]
cv2.imshow('Result1', img1)
cv2.waitKey(0)

