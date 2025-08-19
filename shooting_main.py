import laserControl as lc
import picamFunctions as pf
import numpy as np
import time
import threading
import libe1701py
import shootingFunctions as sf
import cv2
import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class TimeoutWatchdog:
    def __init__(self, timeout_sec, action):
        """
        self : How you name it
        timeout_sec : time you wait before stopping execution
        action : what you call after timeout_sec has run out
        """
        self.timeout = min(timeout_sec, 60) # No matter what, timeout will not be higher than  10 seconds
        self.action = action
        self.thread = threading.Thread(target=self._watchdog, daemon=True)
        self.cancelled = False

    def _watchdog(self):
        time.sleep(self.timeout)
        if not self.cancelled:
            print("Exited due to Timeout Watchdog")
            self.action()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled = True


if __name__ == "__main__":
    
    unit = "pixels"  # "pixels" or "mm"

    pattern_name = "several_sandglasses" #Supported patterns: square, spiral, sandglass, zigzag, several_spirals, "several_sandglasses"

    # Initialize the camera
    picam2, mtx, dist = pf.init_camera()
    time.sleep(0.5)

    #Scanner parameters
    mark_speed = 1000000 #50000  # Speed for marking in bits per second (max spped = 4294960000)
    freq = 10000//2 # Frequency for laser in Hz
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
                #marker_length = 35 # in mm
                centers, ids,corners, frame = pf.detect_aruco_markers(frame,unit) 
            time.sleep(0.1) 
           
            if ids is not None and len(centers) > 0:

                #watchdog = TimeoutWatchdog(20, lambda: libe1701py.close(cardNum))
                #watchdog.start() # Start watching, everything will stop after timeout_sec
                
                #sf.go_to_aruco(cardNum, corners, unit, pattern_name)

                #sf.shoot_target_by_priority_px(cardNum,picam2)

                #Live tracking laser aruco marker
                track_id = 31 # 32  #34
                track_vel_id = 0 

                #sf.live_tracking_px_predict(cardNum, picam2, track_id)

                #sf.live_tracking_px_v2(cardNum, picam2, track_id, track_vel_id)

                sf.live_multitracking_px(cardNum, picam2, track_id, track_vel_id)


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
        #watchdog.cancel() # Stop watching
        sf.close_all_devices(cardNum, picam2)
        print("Exiting the program.")



