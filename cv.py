import cv2
import os
file_name = os.path.join(os.path.dirname(__file__), 'test.jpg')
assert os.path.exists(file_name)

image = cv2.imread("objects.jpg")

blurred = cv2.GaussianBlur(image, (3, 3), 0)

edged = cv2.Canny(blurred, 10, 100)

cv2.imshow("Original image", image)
cv2.imshow("Edged image", edged)
cv2.waitKey(0)
# find the contours in the edged image
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
print(len(contours), "objects were found in this image.")

cv2.imshow("Edged image", edged)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)