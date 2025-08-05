import cv2
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
import yaml
import time


def init_camera():
    """Initialize the camera and return the Picamera2 instance."""
    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": ( 2028,1520)},  # 120 fps 1332x990 resolution / 2028×1080 50 fps, 2028×1520 40 fps
        buffer_count=1
    )

    picam2.configure(config)
    picam2.start()
    return picam2, mtx, dist


def detect_laser_spot(image):
    """Return (x, y) pixel coordinates of the brightest spot in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

    return max_loc, max_val


def detect_aruco_markers(frame,unit = "mm",  mtx=0, dist=0, marker_length=24.0):
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
        
    return centers, ids,corners, frame


def get_marker_homography(corners, marker_length_mm):
    """Returns homography from image to marker real-world (mm) coordinates."""
    # corners[0] shape: (4, 2), 4 corners of marker in image
    img_pts = corners[0].astype(np.float32)

    obj_pts = np.array([
        [0, marker_length_mm],     # Coin inférieur gauche (nouveau (0,0))
        [marker_length_mm, marker_length_mm],  # Coin inférieur droit
        [marker_length_mm, 0],     # Coin supérieur droit
        [0, 0]                     # Coin supérieur gauche
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, obj_pts)
    return H

def pixel_to_mm(pt, H):
    """Convert a pixel (x, y) to world (X, Y) in mm using homography."""
    px = np.array([ [pt[0], pt[1], 1] ], dtype=np.float32).T
    world = H @ px
    world /= world[2]
    return float(world[0]), float(world[1])

def get_spot_coordinates(frame, corners, marker_length_mm, ids, reference_id=23):
    """Returns the real-world coordinates of the laser spot based on a specific marker id."""
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == reference_id: #Remove this if we don't target only one target id
                H = get_marker_homography(corners[i], marker_length_mm)
                p,max_val = detect_laser_spot(frame)
                px = p[0]
                py = p[1]
                if px is not None and py is not None:
                    X_mm, Y_mm = pixel_to_mm((px, py), H)
                    print(f"Spot pixel: ({px}, {py}) → Real-world: ({X_mm:.2f}, {Y_mm:.2f}) mm")
                    
                    #cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
                    #aruco.drawDetectedMarkers(frame, corners, ids)
                    return X_mm, Y_mm, px, py, max_val
                else:
                    print("No laser spot detected.")
                    return 0., 0., 0, 0, 0
        print(f"Marker ID {reference_id} not found.")
    else:
        print("No markers detected.")
        return 0., 0., 0, 0, 0

    # If no marker found or no laser spot detected
    return 0., 0., 0, 0, 0

def get_spot_coordinates_pixels(frame):
    """Returns the pixel coordinates of the laser spot in the frame."""
    p, max_val = detect_laser_spot(frame)
    if p is not None:
        px = p[0]
        py = p[1]
        print(f"Spot pixel: ({px}, {py})")
        return px, py, max_val
    else:
        print("No laser spot detected.")
        return 0, 0, 0



def get_relative_marker_centers(frame, mtx, dist, reference_id, marker_length_mm=24.0):
    """
    Returns the positions of ArUco marker centers relative to the reference marker’s coordinate system.

    Parameters:
    - frame: Input image (RGB, e.g., from Picamera2).
    - mtx: Camera matrix.
    - dist: Distortion coefficients.
    - reference_id: ID of the reference marker.
    - marker_length_mm: Side length of the marker in millimeters (default: 24.0 mm).

    Returns:
    - relative_centers: List of tuples [(x_mm, y_mm), ...] of centers in the reference marker’s coordinate system.
    - ids: Array of detected marker IDs.
    - frame: Annotated image with detected markers.
    """
    # Detect ArUco markers
    _, ids, corners, frame = detect_aruco_markers(frame, "mm", mtx, dist, marker_length_mm)
    
    relative_centers = []
    
    if ids is not None:
        # Find the index of the reference marker
        ref_index = None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == reference_id:
                ref_index = i
                break
        
        if ref_index is not None:
            # Compute the homography for the reference marker
            H = get_marker_homography(corners[ref_index], marker_length_mm)
            
            # Iterate through all detected markers
            for i in range(len(ids)):
                marker_corners = corners[i][0]
                center_px = np.mean(marker_corners, axis=0)  # [x, y] in pixels
                
                # Convert pixel coordinates to millimeters using the homography
                x_mm, y_mm = pixel_to_mm(center_px, H)
                relative_centers.append((x_mm, y_mm))
                
                print(f"Marker {ids[i][0]}: Center at ({x_mm:.2f}, {y_mm:.2f}) mm relative to marker {reference_id}")
        else:
            print(f"Reference marker with ID {reference_id} was not detected.")
    else:
        print("No markers detected.")
    
    return relative_centers, ids, frame



def show_spot(mtx,dist, picam2,unit="mm", marker_length_mm=24.0, reference_id=23):
    # Display the laser spot position on the camera feed depending on the reference ID (in mm)
    frame = picam2.capture_array()
    centers,ids,corners,frame = detect_aruco_markers(frame,unit, mtx, dist, marker_length_mm)
    spotx_mm, spoty_mm, px, py, _ = get_spot_coordinates(frame, corners, marker_length_mm, ids, reference_id)
    print(f"Laser spot at: ({spotx_mm:.2f}, {spoty_mm:.2f}) mm → Pixel: ({px}, {py})")
    cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 255), 2)
    cv2.imshow('Spot detection', frame)
    cv2.waitKey(50)  # Pause to observe


def live_cam(picam):
    """Display live camera feed."""

    # Display the live camera feed
    print("🟢 Press 'q' to quit")
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


