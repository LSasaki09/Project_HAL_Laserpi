import cv2
import numpy as np
import yaml
from picamera2 import Picamera2
import picamFunctions as pf
import matplotlib.pyplot as plt



def detect_plants_prototype(frame):
    '''Detect plants in the frame using HSV color space and return the processed frame with bounding boxes.'''
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Green range (adjust if needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = x + w//2, y + h//2
            cv2.putText(frame, f"Plant ({cx},{cy})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def detect_plants(frame, show = False):
    ExG = 2*frame[:, :, 1] - frame[:, :, 0] - frame[:, :, 2] # ExG formula
    ExG_norm = cv2.normalize(ExG, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalize
    #ExG_norm = cv2.GaussianBlur(ExG_norm, (3,3), 0)
    _, mask = cv2.threshold(ExG_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu thresholding
    kernel = np.ones((2,2), np.uint8)  # structuring element

    # Opening = erosion then dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilation to try to connect disconnected parts
    #mask = cv2.dilate(mask, kernel, iterations=1)

    # Closing = dilation then erosion
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = (mask*255).astype(np.uint8)
    #mask = cv2.bitwise_not(mask)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    centers = np.empty((0, 2), dtype=int)
    rect = np.empty((0, 4), dtype=int) # (x, y) of the top left corner of rectangle. w, h width and height

    for c in contour:
        area = cv2.contourArea(c)
        if area > 3000:  # minimum area threshold in pixels
            perimeter = cv2.arcLength(c, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(c, 30, True)
            approx_perim = cv2.arcLength(approx, True)
            ratio = perimeter / approx_perim
            if ratio < 5:
                x, y, w, h = cv2.boundingRect(c)
                rect = np.append(rect, [[x, y, w, h]], axis = 0)
                center = np.array([[x + w//2, y + h//2]])
                centers = np.append(centers, center, axis = 0)
                # Testing purposes
                if show == True:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.drawContours(mask, [c], -1, (255, 0, 0), 3)
                    for center in centers:
                        cv2.circle(mask, tuple(center), radius=5, color=(255, 0, 255), thickness=-1)
    
    # Testing purposes
    if show == True:
        while show:
            cv2.imshow("Image", frame)
            cv2.imshow("Mask", mask)  # mask is 0/1, multiply so it's visible

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return centers, rect

def bound_rectangles(mask, show = False):
        """
        From a proper mask, will find contour and then bind rectangles to it
        returns the center of the rectangles as well as (x, y) top-left corner and (w, h) width-heigth of the rectagle
        """
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = np.empty((0, 2), dtype=int)
        rect = np.empty((0, 4), dtype=int)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Convert fomr gray to colour

        for c in contour:
            x, y, w, h = cv2.boundingRect(c)
            rect = np.append(rect, [[x, y, w, h]], axis = 0)
            center = np.array([[x + w//2, y + h//2]])
            centers = np.append(centers, center, axis = 0)
            # Testing purposes
            if show == True:
                cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.drawContours(mask, [c], -1, (255, 0, 0), 3)
                for center in centers:
                    cv2.circle(mask, tuple(center), radius=5, color=(255, 0, 255), thickness=-1)
        
        # Testing purposes
        if show == True:
            while show:
                cv2.imshow("Image", frame)
                cv2.imshow("Mask", mask)
                # Quit with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return centers, rect

if __name__ == "__main__":
    #picam2, mtx, dist = pf.init_camera()

    
    # Capture frame
    #frame = picam2.capture_array()
    frame = cv2.imread("/home/pi/projects/Project_HAL_Laserpi/plants_images/test_image3.jpg")
    # Undistort using calibration
    #frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # Detect plants
    centers, rect = detect_plants(frame, show = True)

    print("Now printing centers :")
    print(centers)
    print("Centers printed.")
    print("Now printing rectangles :")
    print(rect)
    print("Rectangles printed.")
    
    #picam2.stop()
    cv2.destroyAllWindows()