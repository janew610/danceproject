import cv2 


# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture("resources/dance.mp4") # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)

# Create video writer
fps = vid_capture.get(5)
video_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter("analyzevideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (video_width, video_height))



# Check to see if the vid_capture fails
if(vid_capture.isOpened() == False):
    print("Error opening the video file")

# While the video capture is running then repeat...
while(vid_capture.isOpened()):

    result, frame = vid_capture.read()

    if result == True:
        height = frame.shape[0]
        width = frame.shape[1]
        frame = cv2.resize(frame, (width, height))

        frame = cv2.flip(frame, 0)



        
        out.write(frame)

        cv2.imshow('video',frame)
        Key = cv2.waitKey(20)
        if Key == ord('q'):
            break
    else:
        break



# Release the video capture object
vid_capture.release()
out.release()
cv2.destroyAllWindows()
