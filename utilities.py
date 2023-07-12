import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720

def save_file(video_capture, file_name):
    fps = video_capture.get(5)
    return cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (WIDTH, HEIGHT))

def resize_img(image):
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image

def overlay(frame, angle1, angle2):
    overlay_image = cv2.imread('samples/resources/human.png')
    overlay_image = cv2.resize(overlay_image, (overlay_image.shape[1]//4,overlay_image.shape[0]//4))

    right_arm = True
    difference = np.abs(angle1[0] - angle2[0])
    if difference > 40:  
        cv2.circle(overlay_image, (137,112), 10, (0,0,255), -1)
        right_arm = False
    
    difference = np.abs(angle1[1] - angle2[1])
    if difference > 40:  
        cv2.circle(overlay_image, (56,112), 10, (0,0,255), -1)

    difference = np.abs(angle1[2] - angle2[2])
    if difference > 40:  
        cv2.circle(overlay_image, (115,198), 10, (0,0,255), -1)
    
    difference = np.abs(angle1[3] - angle2[3])
    if difference > 40: 
        cv2.circle(overlay_image, (76,198), 10, (0,0,255), -1)


    height = overlay_image.shape[0]
    width = overlay_image.shape[1]

    # # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
    frame[0:height, 0:width,:] = overlay_image[0:height, 0:width,:]

    return frame, right_arm