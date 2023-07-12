import cv2
import mediapipe as mp
import numpy as np

image = cv2.imread("resources/body_sample.jpg")

cv2.imshow("Image", image)
cv2.waitKey(0)

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# RGB image
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze image using mediapipe
results = pose.process(imageRGB)

print(results)
print(results.pose_landmarks) # Collection of body part locations
print(results.pose_landmarks.landmark[0]) # List of body part and their information

x = results.pose_landmarks.landmark[0].x
y = results.pose_landmarks.landmark[0].y


height = image.shape[0]
width = image.shape[1]

x_nose = int(x * width)
y_nose = int(y * height)

print(x_nose, y_nose) # Coordinates of nose

# Draw a circle on the nose
circle_center = (x_nose, y_nose)
radius = 50
thickness = 3

cv2.circle(image,circle_center, radius, thickness) 

# Display Image with circle on nose
cv2.imshow("circle", image)
cv2.waitKey(0)


def getBodyPlace(image, results, bodyPart):

    height = image.shape[0]
    width = image.shape[1]

    x_place = int(results.pose_landmarks.landmark[bodyPart].x * width)
    y_place = int(results.pose_landmarks.landmark[bodyPart].y * height)

    return (x_place, y_place)


cv2.circle(image, getBodyPlace(image, results, 24), 10, (0,0,255), 3)
cv2.circle(image, getBodyPlace(image, results, 26), 10, (0,0,255), 3)
cv2.circle(image, getBodyPlace(image, results, 28), 10, (0,0,255), 3)
cv2.imshow("test", image)
cv2.waitKey(0)

def getBodyAngle(results, body1, body2, body3): 
    x_body1 = results.pose_landmarks.landmark[body1].x 
    y_body1 = results.pose_landmarks.landmark[body1].y 

    x_body2 = results.pose_landmarks.landmark[body2].x
    y_body2 = results.pose_landmarks.landmark[body2].y

    x_body3 = results.pose_landmarks.landmark[body3].x
    y_body3 = results.pose_landmarks.landmark[body3].y

    radians = (np.arctan2((y_body3 - y_body2),(x_body3 - x_body2))) - (np.arctan2((y_body1 - y_body2),(x_body1 - x_body2)))
    degree = np.abs(radians * 180 / np.pi)

    if degree > 180.0:
        degree = 360 - degree


    return round(degree)

left_knee_angle = getBodyAngle(results, 24, 26, 28)

print(left_knee_angle)
cv2.waitKey()


cv2.putText(image, str(getBodyAngle(results, 24, 26, 28)), getBodyPlace(image, results, 26), 1, 0.5, (0,0,255))
cv2.imshow("Degree", image)
cv2.waitKey(0)



