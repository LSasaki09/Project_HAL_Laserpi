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

    pattern_name = "several_sandglasses" #Supported patterns: square, spiral, sandglass, zigzag, several_spirals, "several_sandglasses"

    # Initialize the camera
    picam2, mtx, dist = pf.init_camera()
    time.sleep(0.5)

    #Scanner parameters
    mark_speed = 500000 #50000  # Speed for marking in bits per second (max spped = 4294960000)
    freq = 10000//8 # Frequency for laser in Hz
    jump_speed = 4294960000 // 10 # Jump speed in bits per second
    port_name="/dev/ttyACM0"

    # Initialize the laser
    if unit == "mm":
        corr_file="corr_file_v4.bco"
        cardNum = lc.init_laser(port_name, freq, jump_speed, mark_speed,
                                corr_file)
        marker_length_mm = 24.0  # mm (2.4 cm)
        reference_id = 23  # ID of the reference marker to detect

    elif unit == "pixels":
        corr_file="corr_file_v5.bco"
        cardNum = lc.init_laser(port_name, freq, jump_speed, mark_speed,
                                corr_file)
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
                marker_length = 35 # in mm
                centers, ids,corners, frame = pf.detect_aruco_markers(frame,unit,0,0,marker_length)
            time.sleep(0.1) 

            if ids is not None and len(centers) > 0:

                #sf.go_to_several_points_without_cam(cardNum,centers,unit,pattern_name,"scanner_camera_map.csv")
                
                #sf.go_to_aruco(cardNum, corners, unit, pattern_name)

                #Live tracking laser aruco marker
                track_id = 32  #31 #34
                #sf.live_tracking_px(cardNum,picam2,track_id)
                #sf.live_tracking_px_predict(cardNum, picam2, track_id)

                #sf.live_tracking_px_v2(cardNum, picam2, track_id)
                sf.shoot_target_by_priority_px(cardNum,picam2)
                #libe1701py.close(cardNum)

                print("Laser moved to detected marker positions.")
                #cv2.imshow("Laser shooting", frame)
                break
            else:
                print("No markers detected.")

            if  0xFF == ord('q'):
                sf.close_all_devices(cardNum, picam2)
                exit(0)  # Exit if 'q' is pressed
                break

            time.sleep(0.1) 
        
    finally:
        sf.close_all_devices(cardNum, picam2)
        print("Exiting the program.")



