import laserControl as lc
import picamFunctions as pf
import numpy as np
import time
import threading
import libe1701py
import shootingFunctions as sf
import cv2
import joblib


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

    pattern_name = "sandglass" #Supported patterns: square, spiral, sandglass, zigzag, several_spirals, "several_sandglasses"

    # Initialize the camera
    #picam2, mtx, dist = pf.init_camera()

    #Scanner parameters
    mark_speed = 2000000 #50000  # Speed for marking in bits per second (max spped = 4294960000)
    freq = 10000//5 #35 # Frequency for laser in Hz
    jump_speed = 4294960000 // 10 # Jump speed in bits per second
    port_name="/dev/ttyACM0"
    csv_path = "scanner_camera_map.csv"


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
        track_vel_id = 0 

        sf.live_multitracking_multiprocess_px(cardNum, track_vel_id, csv_path)

        
    finally:
        #watchdog.cancel() # Stop watching
        #sf.close_all_devices(cardNum, picam2)
        libe1701py.close(cardNum)
        print("Exiting the program.")



