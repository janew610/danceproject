import cv2

image = cv2.imread("resources/dance_sample.jpg")

# Resize
height = image.shape[0]
width = image.shape[1]
image = cv2.resize(image, (width//3, height//3))

cv2.imshow("Hello", image)
cv2.waitKey(0)
