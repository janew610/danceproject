import cv2
import mediapipe as mp

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

# Draw landmarks onto the given image "imageRGB"
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


cv2.imshow("Image", image)
cv2.waitKey(0)