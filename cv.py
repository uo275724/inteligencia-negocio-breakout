import cv2
import os
import numpy as np
file_name = os.path.join(os.path.dirname(__file__), 'test.jpg')
assert os.path.exists(file_name)

image = cv2.imread(file_name)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define the lower and upper bounds of the color that you want to detect
lower_color = np.array([0, 0, 
0])
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

cv2.imshow("Edged image", mask)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)
# Going through every contours found in the image. 
for cnt in contours : 

    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 

    # draws boundary of contours. 
    cv2.drawContours(image, [approx], 0, (0, 0, 255), 5) 

    # Used to flatted the array containing 
    # the co-ordinates of the vertices. 
    n = approx.ravel() 
    i = 0

    for j in n : 
        if(i % 2 == 0): 
            x = n[i] 
            y = n[i + 1] 

            # String containing the co-ordinates. 
            string = str(x) + " " + str(y) 
            '''
            if(i == 0): 
                # text on topmost co-ordinate. 
                cv2.putText(image, "Arrow tip", (x, y), 
                                font, 0.5, (255, 0, 0)) 
            else: 
                # text on remaining co-ordinates. 
                cv2.putText(image, string, (x, y), 
                        font, 0.5, (0, 255, 0)) 
            '''
        i = i + 1

# Showing the final image. 
cv2.imshow('image2', image) 

# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()