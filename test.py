from DanceAnalyzer import danceanalyzer
import cv2
import numpy as np

# tool1 = danceanalyzer()
# # file_location, csv_location = tool1.analyzeVideo('samples/resources/pose.mp4')
# tool1.liveWebCam('samples/resources/pose_output.mp4', 'samples/resources/pose_output.csv')








overlay_image = cv2.imread('samples/resources/human.png')
overlay_image = cv2.resize(overlay_image, (overlay_image.shape[1]//4,overlay_image.shape[0]//4))

print(overlay_image.shape)

cv2.circle(overlay_image, (115,198), 10, (0,0,255), -1)

cv2.imshow("image", overlay_image)
cv2.waitKey(0)































