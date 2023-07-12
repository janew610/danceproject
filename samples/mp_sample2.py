import cv2
import mediapipe as mp

# Setting up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_tool = mp_pose.Pose()

image = cv2.imread("pose.webp")

# Convert the default image into a RGB format
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze the image using mediapipe
result = pose_tool.process(imageRGB)

# Draws all the lines in the imageRGB
mp_drawing.draw_landmarks(imageRGB, result.pose_landmarks, mp_pose.POSE_CONNECTIONS) 

# Display imageRGB
cv2.imshow("pose image", imageRGB)
cv2.waitKey(0)



