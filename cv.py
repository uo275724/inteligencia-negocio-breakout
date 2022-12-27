import cv2
import os
import numpy as np
def test():
    file_name = os.path.join(os.path.dirname(__file__), 'test.png')
    assert os.path.exists(file_name)

    image = cv2.imread(file_name)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the color that you want to detect
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([150, 145, 140])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    #blurred = cv2.GaussianBlur(mask, (3, 3), 0)

    #edged = cv2.Canny(mask, 10, 100)
    font = cv2.FONT_HERSHEY_COMPLEX 
    # find the contours in the edged image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")
    # Going through every contours found in the image. 
    for cnt in contours : 

        approx = cv2.approxPolyDP(cnt, 0.000001 * cv2.arcLength(cnt, True), True) 
        #0.009
        # draws boundary of contours. 
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 5) 
        # Used to flatted the array containing 
        # the co-ordinates of the vertices. 
        n = approx.ravel() 
        i = 0
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # String containing the co-ordinates.  
        # putting shape name at center of each shape
        string = str(cX) + " " + str(cY) 
        
        if len(approx) != 4:
            print("Ball: " + string)
        else:
            print("Paddle: " + string)
        

    # Showing the final image. 
    cv2.imshow('image2', image) 

    # Exiting the window if 'q' is pressed on the keyboard. 
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
def getCoordinates(view):
    font = cv2.FONT_HERSHEY_COMPLEX 
    image = view.transpose([1, 0, 2])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the color that you want to detect
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([140, 135, 130]) 
    mask = cv2.inRange(hsv, lower_color, upper_color)
    #blurred = cv2.GaussianBlur(mask, (3, 3), 0)

    #edged = cv2.Canny(mask, 10, 100)
    font = cv2.FONT_HERSHEY_COMPLEX 
    # find the contours in the edged image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

    print(len(contours), "objects were found in this image.")
    # Going through every contours found in the image. 
    paddleX = 0
    ballX = 0
    ballY = 0
    for cnt in contours : 

        approx = cv2.approxPolyDP(cnt, 0.000001 * cv2.arcLength(cnt, True), True) 

        # Used to flatted the array containing 
        # the co-ordinates of the vertices. 
        n = approx.ravel() 
        i = 0
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # String containing the co-ordinates.  
        # putting shape name at center of each shape
        
        
        if len(approx) != 4:
            ballX = cX-9
            ballY = cY-9
            cv2.putText(image_copy, "Ball ({},{})".format(ballX,ballY), (ballX, ballY-5),font, 0.5, (255, 0, 0))
        else:
            paddleX = cX-46
            paddleY = cY
            cv2.putText(image_copy, "Paddle({},{})".format(paddleX,paddleY), (paddleX, paddleY-15),font, 0.5, (255, 0, 0))
    # Showing the final image. 
    cv2.imshow('image2', image_copy) 

    # Exiting the window if 'q' is pressed on the keyboard. 
    cv2.waitKey(1) 
    return [paddleX,ballX,ballY]

if __name__ == "__main__":
    test()