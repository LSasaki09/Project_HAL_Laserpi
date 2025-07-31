import cv2
from cv2 import aruco
import yaml
import numpy as np
from picamera2 import Picamera2

# Load the calibration parameters from the calibration.yaml file
with open('calibration.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
mtx = np.array(loadeddict.get('camera_matrix'))
dist = np.array(loadeddict.get('dist_coeff'))

# Initialize the camera with Picamera2
picam2 = Picamera2()
picam2.start()

# Define the ArUco dictionary (e.g., DICT_6X6_250)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Size of the marker in cm (adjust according to your markers)
marker_length = 2.4 #3.5  # in centimeters

while True:
    # Capture an image as a numpy array
    frame = picam2.capture_array()
    # Convert from RGB (Picamera2) to BGR (OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate the pose of each detected marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        # Draw the detected markers and their axes
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            frame = cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], marker_length / 2)

            # Coordinates of the marker's center (in cm if marker_length is in cm)
            center = tvecs[i][0]
            #print(f"Marker {ids[i][0]}: Center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) cm")

    # Display the image
    cv2.imshow('ArUco Detection', frame)

    # Quit with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()