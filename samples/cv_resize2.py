import cv2

image = cv2.imread("pose.webp")
cv2.imshow("orginal image", image)

down_width = 200
down_height = 100
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points)


cv2.imshow("smaller blue", resized_down)

cv2.waitKey(0)
