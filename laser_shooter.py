import laserControl as lc
import picamFunctions as pf
import numpy as np
import time
import libe1701py
import scannerFunctions as sf
import cv2
import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    
    unit = "pixels"  # "pixels" or "mm" 

    pattern_name = "spiral" #Supported patterns: square, spiral, sandglass, zigzag

    # Initialize the camera
    picam2, mtx, dist = pf.init_camera()
    time.sleep(1)

    # Initialize the laser
    if unit == "mm":
        cardNum = lc.init_laser(port_name="/dev/ttyACM0", freq=10000, jump_speed=4294960000//10, mark_speed=50000,
                                corr_file="corr_file_v4.bco")
        marker_length_mm = 24.0  # mm (2.4 cm)
        reference_id = 23  # ID of the reference marker to detect

    elif unit == "pixels":
        cardNum = lc.init_laser(port_name="/dev/ttyACM0", freq=10000, jump_speed=4294960000//10, mark_speed=4294960000//100000,
                                corr_file="corr_file_v5.bco")
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        exit(1)

    try:
        while True:
            frame = picam2.capture_array()
            if unit == "mm":
                # Detect ArUco markers and get their centers in mm
                centers, ids, frame = pf.get_relative_marker_centers(frame, mtx, dist, reference_id, 
                                                                                marker_length_mm)
            elif unit == "pixels":
                centers, ids,corners, frame = pf.detect_aruco_markers(frame,unit)
            time.sleep(0.1) 

            if ids is not None and len(centers) > 0:

                #sf.go_to_several_points_with_cam(cardNum,centers,mtx,dist,picam2,marker_length_mm, reference_id, "scanner_camera_map.csv")

                #sf.go_to_several_points_without_cam(cardNum,centers,unit,pattern_name,"scanner_camera_map.csv")
                
                #sf.go_to_aruco(cardNum, corners, unit, pattern_name)

                print("Laser moved to detected marker positions.")
                #cv2.imshow("Laser shooting", frame)
                break
            else:
                print("No markers detected.")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1) 
        
    finally:
        sf.close_all_devices(cardNum, picam2)
        print("Exiting the program.")



