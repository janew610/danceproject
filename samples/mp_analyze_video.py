import cv2 
import mediapipe as mp
import numpy as np



def analyzepic(image, mp_drawing, mp_pose, pose):

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)                   
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image, results

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

def getBodyPlace(image, results, bodyPart):

    height = image.shape[0]
    width = image.shape[1]

    x_place = int(results.pose_landmarks.landmark[bodyPart].x * width)
    y_place = int(results.pose_landmarks.landmark[bodyPart].y * height)

    return (x_place, y_place)

def drawAllAngle(image, results):
    cv2.putText(frame, str(getBodyAngle(results, 12, 14, 16)), getBodyPlace(frame, results, 14), 1, 2, (0,0,255)) # Right Elbow
    cv2.putText(frame, str(getBodyAngle(results, 11, 13, 15)), getBodyPlace(frame, results, 13), 1, 2, (0,0,255)) # Left Elbow
    cv2.putText(frame, str(getBodyAngle(results, 24, 26, 28)), getBodyPlace(frame, results, 26), 1, 2, (0,0,255)) # Right knee
    cv2.putText(frame, str(getBodyAngle(results, 23, 25, 27)), getBodyPlace(frame, results, 25), 1, 2, (0,0,255)) # Left knee


# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture("resources/dance.mp4") # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)

# Create Video Writer Object
fps = vid_capture.get(5)
video_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter("analyzevideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (video_width, video_height))


# Setup Mediapipe in the beginnings
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


if (vid_capture.isOpened() == False):
    print("Error opening the video file")

while(vid_capture.isOpened()):
    # vid_capture.read() methods returns two values, first element is a boolean
    # and the second is frame
    result, frame = vid_capture.read()
    if result == True:
        # Resize Frame
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.resize(frame, (width, height))

        frame, results = analyzepic(frame, mp_drawing, mp_pose, pose)
        
        drawAllAngle(frame, results)

        out.write(frame)
        cv2.imshow('Frame',frame)
        # 0 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
out.release()
cv2.destroyAllWindows()

