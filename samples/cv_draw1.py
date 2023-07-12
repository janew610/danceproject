# Import dependencies
import cv2

# Read the image
img = cv2.imread('sample.jpg')
#Display the input image
cv2.imshow('Original Image',img)
cv2.waitKey(0)

#Make copy of the image
imageLine = img.copy()
# Draw the image from point A to B
pointA = (200,80)
pointB = (450,200)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)


# Make a copy of image
imageCircle = img.copy()

imgHeight = img.shape[0]
imgWidth = img.shape[1]

# define the center of circle
circle_center = (imgWidth//2, imgHeight//2)
# define the radius of the circle
radius =100
#  Draw a circle using the circle() Function
cv2.circle(imageCircle, circle_center, radius, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA) 
# Display the result
cv2.imshow("Image Circle",imageCircle)
cv2.waitKey(0)


