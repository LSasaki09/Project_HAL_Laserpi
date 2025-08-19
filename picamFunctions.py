import cv2
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
import yaml
import time


def init_camera(ExposureTime = 66657, gain = 8.0, resolution = [2028,1520]):
    """Initialize the camera and return the Picamera2 instance."""
    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (resolution[0], resolution[1])},  # 120 fps 1332x990 resolution / 2028×1080 50 fps, 2028×1520 40 fps (3mp), 2560x1920 (5MP), 4056x3040(max res) 
        buffer_count=1
    )

    picam2.configure(config)
    # Disable auto exposure
    picam2.set_controls({"AeEnable": False})
    # Set manual exposure and gain
    picam2.set_controls({"ExposureTime": ExposureTime,"AnalogueGain": gain})
    picam2.start()
    return picam2, mtx, dist

def modify_exposure_and_gain(picam2, exposure = 66657, gain = 8.0, debug = False):
    picam2.stop()
    time.sleep(0.5)
    # Disable auto exposure
    picam2.set_controls({"AeEnable": False})
    # Set manual exposure and gain
    picam2.set_controls({"ExposureTime": exposure,"AnalogueGain": gain})
    picam2.start()
    time.sleep(0.5)
    if debug == True:
        metadata = picam2.capture_metadata()
        print("Applied ExposureTime:", metadata.get("ExposureTime"))
        print("Applied AnalogueGain:", metadata.get("AnalogueGain"))

def detect_laser_spot(image):
    """Return (x, y) pixel coordinates of the brightest spot in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

    return max_loc, max_val

def detect_laser_spot2(image, test = False):
    """
    Return [cx, cy] pixel coordinates, spot_detected (boolean) .
    Prefoms gaussian blur then thresholding
    find the best contour in the area and then compute its center of mass
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, 0)
    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)

    # finding contour with maximum area and store it as best_cnt
    spot_detected = False
    max_area = 0
    cx, cy = 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

            # finding centroids of best_cnt and draw a circle there
            M = cv2.moments(best_cnt)
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    if max_area > 0:
        spot_detected = True

    if test == True:
        max_pos, _ = detect_laser_spot(image)
        cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
        # Before drawing
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_color[cy, cx] = (0, 0, 255)
        thresh_color[max_pos[1], max_pos[0]] = (0, 255, 0)
        blurred_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        blurred_color[cy, cx] = (0, 0, 255)
        blurred_color[max_pos[1], max_pos[0]] = (0, 255, 0)
        if (cx == max_pos[0]) and (cy == max_pos[1]):
            thresh_color[cy, cx] = (255, 0, 0)
            blurred_color[cy, cx] = (255, 0, 0)
        #small = cv2.resize(thresh_color, (1280, 720))
        #small2 = cv2.resize(blurred_color, (1280, 720))
        #cv2.imshow("Contoured Image", small)
        cv2.imshow("Contoured Image", blurred_color)
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  # spacebar
                break
        cv2.destroyAllWindows()
    
    return [cx, cy], spot_detected

def detect_aruco_markers(frame,unit = "pixels",  mtx=0, dist=0, marker_length=24.0):
    """Detect ArUco markers in the frame and return their centers."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    centers = []

    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if unit == "mm":
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
        for i in range(len(ids)):
            if unit == "mm":
                frame = cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], marker_length / 2)
                center = tvecs[i][0][:2]  # Extraire (x_mm, y_mm), ignorer z_mm
                centers.append(center)
                #print(f"Marker {ids[i][0]}: Center at ({center[0]:.2f}, {center[1]:.2f}) mm")
            elif unit == "pixels":
                # Calculate the center of the marker in pixels
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                centers.append((center_x, center_y))
                #print(f"Marker {ids[i][0]}: Center at ({center_x:.2f}, {center_y:.2f}) pixels")

    return centers, ids, corners, frame



def get_spot_coordinates_pixels(frame):
    """Returns the pixel coordinates of the laser spot in the frame."""
    p, max_val = detect_laser_spot(frame)
    if p is not None:
        px = p[0]
        py = p[1]
        #print(f"Spot pixel: ({px}, {py})")
        return px, py, max_val
    else:
        print("No laser spot detected.")
        return 0, 0, 0


def live_cam(picam):
    """Display live camera feed."""

    # Display the live camera feed
    print(" Press 'q' to quit")
    while True:
        frame = picam.capture_array()
        small = cv2.resize(frame, (640, 480)) 
        cv2.imshow("Live Camera Feed", small)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    picam.stop()
    cv2.destroyAllWindows()


def aruco_detection(picam2, mtx, dist):
    """Detect ArUco markers using Picamera2."""
    # Define the ArUco dictionary (e.g., DICT_6X6_250)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
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
                print(f"Marker {ids[i][0]}: Center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) cm")
                time.sleep(0.2)  # Pause to observe the marker detection

        # Display the image
        small = cv2.resize(frame, (640, 480))  
        cv2.imshow('ArUco Detection', small)

        # Quit with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    picam2.stop()
    cv2.destroyAllWindows()