import cv2 
import mediapipe as mp

def analyzepic(image, mp_drawing, mp_pose, pose):


    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)                   
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture("resources/dance.mp4") # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)

# Setup Mediapipe in the beginning
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

        word1 = "Hello"
        number1 = 12
        boolean1 = True

        frame = analyzepic(frame, mp_drawing, mp_pose, pose)
        
        cv2.imshow('Frame',frame)
        # 0 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)
        
        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()