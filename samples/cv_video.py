import cv2 

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(0) # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)

# Check to see if the vid_capture fails
if(vid_capture.isOpened() == False):
    print("Error opening the video file")

# While the video capture is running then repeat...
while(vid_capture.isOpened()):

    result, frame = vid_capture.read()

    if result == True:
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.resize(frame, (width//3, height//3))


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
