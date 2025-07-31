import laserControl as lc
import picamFunctions as pf
import scannerFunctions as sf
import libe1701py
import cv2
import numpy as np
import time



if __name__ == "__main__":

    # Initialize the camera
    picam2, mtx, dist = pf.init_camera()
    time.sleep(2)

    # Initialize the laser
    cardNum = lc.init_laser(port_name="/dev/ttyACM0", freq=10000, jump_speed=4294960000//10, mark_speed=50000)
    marker_length_mm = 24.0  # mm (2.4 cm)
    reference_id = 23  # ID of the reference marker to detect

    # Path to CSV file
    csv_path = "scanner_camera_map.csv"

    '''camera test'''
    # ArUco detection test
    pf.aruco_detection(picam2, mtx, dist)

    #pf.live_cam(picam2)


    ''' Laser spot position test '''
    #square_size = 67108860 
    #lc.test_connection()
    #lc.draw_square(cardNum, square_size, 0, 0) #draw a squrae in bit coordinates



    '''Scanner tests (camera + laser)'''

    #sf.collect_pts_calib_scanner()

    #sf.plot_points(mm_xy, bit_xy)

    #mm_xy, bit_xy, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale = sf.load_data(csv_path)

    #sf.draw_square_world_frame(cardNum, 30, 0 ,0 , mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)





    