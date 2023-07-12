import cv2

blue = cv2.imread("blue.png")

cv2.imshow("blue", blue)

cv2.waitKey(0)

blueline = blue.copy()

cv2.line(blueline, (0,100), (100,100), (51,153,255), 5)

cv2.imshow("blueline", blueline)

cv2.waitKey(0)

bluetext = blue.copy()
cv2.putText(bluetext, "hi", (100,100), 0, 3, (153,51,153) )

cv2.imshow("bluetext", bluetext)
cv2.waitKey(0)