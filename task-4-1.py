# References:
# - https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# - https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# - https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

# Improvements made from the baseline scripts include creating a mask to identify pixels falling inside the defined range of hsv values for black,
# using contours reliant on the mask to find the largest black object, and drawing a rectangle around that specific object
import numpy as np
import cv2 as cv

# Identify largest black object and draw bounding rectangle around it
def track_black():
    capture = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = capture.read()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	# define lower and upper hsv ranges for black
	blk_lower_range = np.array([0, 0, 0])
	blk_upper_range = np.array([180, 255, 50])

	mask = cv.inRange(hsv, blk_lower_range, blk_upper_range)
	contours,hierarchy = cv.findContours(mask, 1, 2)

	largest_contour = max(contours, key=cv2.contourArea)
	rect = cv.minAreaRect(largest_contour)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	cv.drawContours(frame,[box],0,(0,0,255),2)

	cv2.imshow('frame', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

    # release capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_black()
