import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(p1,p2,p3):
    first = np.array(p1)  # information about the first joint 
    second = np.array(p2)
    third = np.array(p3)

    # first[0] <-- x position
    # first[1] <-- y position

    a = np.arctan2((third[1]- second[1]),(third[0]- second[0]))
    b = np.arctan2((first[1]- second[1]),(first[0]- second[0]))

    radians = a - b

    # radian = np.arctan((third[1]- second[1]),(third[0]- second[0])) - np.arctan((first[1]- second[1]),(first[0]- second[0]))
    degree = np.abs(radians * 180 / np.pi)

    if degree > 180.0:
        degree = 360 - degree

    return round(degree)


image = cv2.imread("body_sample.jpg")

# Resize
height = image.shape[0]
width = image.shape[1]
image = cv2.resize(image, (width//2, height//2))

# Display resized image
cv2.imshow("Resized Image", image)
cv2.waitKey(0)

# RGB image
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()   # Object to process landmarks on RGB image

results = pose.process(imageRGB) # Analyze posture and store results
print(results.pose_landmarks) # Contains information of every body part

# Display landmarks
mp_drawing.draw_landmarks(imageRGB, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imshow("Landmarks", imageRGB)
cv2.waitKey(0)


# Gather Left Elbow angle
# collect left shoulder, left elbow, left wrist landmarks (11, 13, 15) or (mp_pose.PoseLandmark.LEFT_SHOULDER, .LEFT_ELBOW, .LEFT_WRIST)
print(results.pose_landmarks.landmark[11])
print(results.pose_landmarks.landmark[11].x)
shoulder = [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y]
elbow = [results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y]
wrist = [results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y]

print("Shoulder", shoulder)
print("Elbow", elbow)
print("Wrist", wrist)

# Calculate angle
angle = calculate_angle(shoulder, elbow, wrist)
print(angle)
cv2.waitKey(0)

# Covert image with landmarks into BGR
imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)

# Draw angle
position = ((int) (elbow[0] * (width//2)), (int) (elbow[1] * (height//2)))
print(type(position))
print(position)

cv2.putText(imageBGR, str(angle), position, cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), thickness=1)
cv2.imshow("Angle Displayed", imageBGR)
cv2.waitKey(0)


# Return landmark results from BGRimage
def mediapipe_results(image, pose):

    # Covert BGR to RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB) # Analyze posture and store results

    return results


def process_image(image, mp_drawing, mp_pose, results):

    # Draw landmarks on image
    mp_drawing.draw_landmarks(imageRGB, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw left elbow angle
    shoulder = [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y]
    elbow = [results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y]
    wrist = [results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y]

    angle = calculate_angle(shoulder, elbow, wrist)

    position = ((int) (elbow[0] * (width//2)), (int) (elbow[1] * (height//2)))
    cv2.putText(image, str(angle), position, cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), thickness=1)

    return image


