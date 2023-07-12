import cv2


video = cv2.VideoCapture("video.mp4")

if(video.isOpened() == False):
    "Error opening video"


while(video.isOpened()):
    
    # video.read() returns the next frame in the video into image
    # result contains if the action was successfull or not
    result, image = video.read()

    if(result == True):

        # Get height and width of current image
        height = image.shape[0]
        width = image.shape[1]

        # Resize the frame
        down_width = width // 4
        down_height = height // 4
        down_points = (down_width, down_height)
        resized_down = cv2.resize(image, down_points)
    
        # Display the current frame
        cv2.imshow("video", resized_down)
        key = cv2.waitKey(30)

        if key == ord('q'):
             break
    else:
        break

# close video
video.release()
cv2.destroyAllWindows()

