import laserControl as lc
import picamFunctions as pf
import shootingFunctions as sf
import libe1701py
import cv2
import numpy as np
import time

#import Tampon_file as tf

def on_off_laser (cardNum):
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")  # Turn on laser
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "1") 
    libe1701py.execute(cardNum)
    print("Laser turned on.")
    time.sleep(2)
    print("Laser turned off.")
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, '0')  # Turn off laser
    #libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.execute(cardNum)
    time.sleep(2)
    #libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "1")  # Turn on laser
    libe1701py.execute(cardNum)
    print("Laser turned on.")
    time.sleep(2)

if __name__ == "__main__":

    # Initialize the camera
    picam2, mtx, dist = pf.init_camera()
    time.sleep(1)

    # Initialize the laser
    cardNum = lc.init_laser(port_name="/dev/ttyACM0", freq=20000, jump_speed=4294960000//4, mark_speed=4000000) # mark_speed=500000
    marker_length_mm = 24.0  # mm (2.4 cm)
    reference_id = 23  # ID of the reference marker to detect
    
    #lc.test_connection()

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


    #on_off_laser(cardNum)
    #sf.shoot_target_by_priority_px(cardNum,picam2)
    #libe1701py.close(cardNum)

    """
    libe1701py.mark_abs(cardNum, 0, 0, 0)  # Mark the current position
    libe1701py.laserControl
    libe1701py.execute(cardNum)
    time.sleep(1)
    libe1701py.mark_abs(cardNum, 0, 0, 0) 
    """

    # test punching
    """
    grid_size=15

    bit_range = 67108860//100

    bit_positions = np.linspace(-bit_range, bit_range, grid_size, dtype=int)
    grid_points = [(x, y) for x in bit_positions for y in bit_positions]

    for idx, (bx, by) in enumerate(grid_points):
        print(f"[{idx+1}/{len(grid_points)}] Point: ({bx}, {by})")
        lp.punching(cardNum, [bx, by])

        print(f"Point ({bx}, {by}) processed.")

    libe1701py.close(cardNum)
    """

    
    #timestamps = tf.measure_communication_delay(cardNum, picam2)
    #timestamps = tf.measure_camera_delay(cardNum, picam2)
    
    # Print the timestamps
    #print("Timestamps for communication delay measurement:")
    #for i, ts in enumerate(timestamps):
     #   print(f"Measurement {i}: {ts:.6f} ")
    

    #sf.close_all_devices(cardNum, picam2)


    '''Scanner tests (camera + laser)'''

    #sf.collect_pts_calib_scanner()

    #sf.plot_points(mm_xy, bit_xy)

    #mm_xy, bit_xy, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale = sf.load_data(csv_path)

    #sf.draw_square_world_frame(cardNum, 30, 0 ,0 , mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)





    