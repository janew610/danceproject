import cv2

# Relevant link: https://learnopencv.com/read-display-and-write-an-image-using-opencv/

# Read a selected file image and store into a variable
react_grey = cv2.imread("images.jpeg", cv2.IMREAD_GRAYSCALE)



# Displays a window with the title "React Image" and the given image
cv2.imshow("React Image", react_grey)

# Pauses the program and waits for the given amount
# if 0 is given then the program pauses until a key is pressed
cv2.waitKey(0)

# Destroys all the windows created
cv2.destroyAllWindows() 


# To save an image use imwrite()
# Requires two inputs: filename it will be saved as, image you want to save
cv2.imwrite("greyscale.jpg", react_grey)

blue = cv2.imread("blue.png")
cv2.imshow("Blue Image", blue)
cv2.waitKey(0)