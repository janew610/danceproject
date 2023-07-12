

'''
1)  import neccessary libraries
2)  select dance video to analyze
        - video can be prerecorded or stream webcam
3)  make sure to set up mediapipe tools

4) check to see if video opened successfully

5) in a while loop start reading each frame of the video
    For every frame:
        - resize the image
        - use mediapipe to analyze image to get results and draw landmarks
        - draw angles onto the joints
        - display the image

6) release the video object

'''

import cv2 
import mediapipe as mp
import numpy as np
from utilities import save_file, resize_img, overlay

class danceanalyzer:
    
    # The class will set up the mediapipe tools to be used in later functions
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def analyzepic(self, image):

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imageRGB)                   
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image, results
    
    def getBodyPlace(self, image, results, bodyPart):

        height = image.shape[0]
        width = image.shape[1]

        x_place = int(results.pose_landmarks.landmark[bodyPart].x * width)
        y_place = int(results.pose_landmarks.landmark[bodyPart].y * height)

        return (x_place, y_place)
        
    def getBodyAngle(self, results, body1, body2, body3): 
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
   
    # Causes an issue when live analyzing the webcam
    def getShoulderSlope(self, results):
    
        x1 = results.pose_landmarks.landmark[12].x
        y1 = results.pose_landmarks.landmark[12].y

        x2 = results.pose_landmarks.landmark[11].x
        y2 = results.pose_landmarks.landmark[11].y

        slope = (y2-y1)/(x2-x1)
        
        if slope > 0:
            print("Tilting left")
        else:
            print("Tilting right")

    def drawAllAngles(self, image, results):
        angleList = []
        right_elbow = self.getBodyAngle(results, 12, 14, 16)
        angleList.append(right_elbow)
        cv2.putText(image, str(right_elbow), self.getBodyPlace(image, results, 14), 1, 2, (0,0,255)) # Right Elbow
        
        left_elbow = self.getBodyAngle(results, 11, 13, 15)
        angleList.append(left_elbow)
        cv2.putText(image, str(left_elbow), self.getBodyPlace(image, results, 13), 1, 2, (0,0,255)) # Left Elbow
        
        right_knee = self.getBodyAngle(results, 24, 26, 28)
        angleList.append(right_knee)
        cv2.putText(image, str(right_knee), self.getBodyPlace(image, results, 26), 1, 2, (0,0,255)) # Right knee
        
        left_knee = self.getBodyAngle(results, 23, 25, 27)
        angleList.append(left_knee)
        cv2.putText(image, str(left_knee), self.getBodyPlace(image, results, 25), 1, 2, (0,0,255)) # Left knee
        
        return angleList

    # Purpose is to take a video, analyze it, and save it to file
    # Make sure that the video size is consistant
    def analyzeVideo(self, video_path):
        print("attempt to analyze and save video")
        vid_capture = cv2.VideoCapture(video_path) # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)
        filename = video_path.split('.')[0] + '_output.mp4'
        video_output = save_file(vid_capture, filename)

        if (vid_capture.isOpened() == False):
            print("Error opening the video file")

        video_angles = []
        while(vid_capture.isOpened()):
            # vid_capture.read() methods returns two values, first element is a boolean
            # and the second is frame
            result, frame = vid_capture.read()
            if result == True:
                # Resize Frame
                frame = resize_img(frame)

                try:
                    # Analyze Video
                    frame, results = self.analyzepic(frame)
                    self.getShoulderSlope(results)
                    frame_angles = self.drawAllAngles(frame, results)
                    video_angles.append(frame_angles)
                except:
                    print('mediapipe was not able to find a body')
                    video_angles.append([-1, -1, -1, -1])

                # Save video
                video_output.write(frame)

            else:
                break
        
        print(video_angles)
        video_angles = np.array(video_angles)
        csv_file = video_path.split('.')[0] + '_output.csv'
        np.savetxt(csv_file, video_angles, delimiter=',')

        # Release the video capture object
        vid_capture.release()
        video_output.release()
        print("analyzed video successfully saved")

        return filename, csv_file

    def liveWebCam(self, video_path, csv_path):

        # Open video 
        webcam = cv2.VideoCapture(1)
        video = cv2.VideoCapture(video_path)

        # Open angle file
        video_angles = np.loadtxt(csv_path, delimiter=",")

        if ((webcam.isOpened() == False) and (video.isOpened() == False)):
            print("Error opening one of the video files")

        rowIndex = 0
        right_elbow_correct = 0
        while(webcam.isOpened() and video.isOpened()):

            # Extract a single frame from each footage
            webcam_success, webcam_frame = webcam.read()
            video_success, video_frame = video.read()

            if webcam_success and video_success:
                try:
                    # Live Analysis of Webcam
                    webcam_frame, landmarks = self.analyzepic(webcam_frame)
                    # self.getShoulderSlope(webcam_frame)
                    webcam_frame_angles = self.drawAllAngles(webcam_frame, landmarks)
                    video_frame_angles = video_angles[rowIndex]

                    # webcam_frame_angles[1] == video_frame_angles[1]


                        
                except:
                    print("error could not read webcam")

                rowIndex = rowIndex + 1

                # Flip webcam frame so that it can reflect the user's movements as a mirror
                webcam_frame = cv2.flip(webcam_frame, 1)

                # Add overlay
                webcam_frame, right_arm_value = overlay(webcam_frame, webcam_frame_angles, video_frame_angles)

                if right_arm_value:
                    right_elbow_correct = right_elbow_correct + 1

                # Combine the two final frames into one image and display
                image = np.concatenate((webcam_frame, video_frame), axis=1)

                cv2.imshow('Frame', image)
                # 0 is in milliseconds, try to increase the value, say 50 and observe
                key = cv2.waitKey(20)

                if key == ord('q'):
                    break
            else:
                break 

        print("Your right elbow was accurate for " + str(right_elbow_correct / rowIndex) + "%")

        webcam.release()
        video.release()
        cv2.destroyAllWindows()

        
