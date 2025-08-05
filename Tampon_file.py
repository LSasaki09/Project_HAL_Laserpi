import laserControl as lc
import picamFunctions as pf
import cv2
from picamera2 import Picamera2
import numpy as np
import time
import libe1701py

def target_order(list_of_targets):
    ordered_list = list_of_targets
    np.sort(ordered_list[0:1], axis = -1)
    return ordered_list



#picam2, mtx, dist = pf.init_camera()
#time.sleep(0.5)
#cardNum = lc.init_laser()  

#libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")#Turn on/off laser
#libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
#libe1701py.execute(cardNum)

def measure_communication_delay(cardNum, picam2):
    time_stamps = np.zeros(100)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
    i = 1

    start = time.perf_counter()
    prev_time = start

    libe1701py.jump_abs(cardNum, 0, 0, 0)

    while i <= 99:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, spot_intensity = pf.detect_laser_spot(frame)
        print(f"Spot intensity: {spot_intensity}")

        if spot_intensity < 230: #means laser turned off

            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, '1') #turn on
            libe1701py.execute(cardNum)
            while spot_intensity < 230:
                print("Waiting for laser to turn on...")
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, spot_intensity = pf.detect_laser_spot(frame)

            #time_stamps[i] = time.perf_counter() - time_stamps[i-1] - start
            current_time = time.perf_counter()
            time_stamps[i] = current_time - prev_time
            prev_time = current_time

        else: #means laser turned on
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, '0') #turn off
            libe1701py.jump_abs(cardNum, 0, 0, 0) #jump to 0,0,0
            libe1701py.execute(cardNum)
            #print("Laser turned off")
            while spot_intensity > 230:
                
                print("Waiting for laser to turn off...")
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, spot_intensity = pf.detect_laser_spot(frame)

           # time_stamps[i] = time.perf_counter() - time_stamps[i-1] - start 
            current_time = time.perf_counter()
            time_stamps[i] = current_time - prev_time
            prev_time = current_time
        i += 1
    
    return time_stamps

