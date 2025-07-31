'''Part of code / functions not used '''


####################### ScannerFunctions #######################

def go_to_several_points_with_cam(cardNum, points_mm, mtx, dist, picam2, marker_length_mm=24., reference_id=23, csv_path="scanner_camera_map.csv"):
    """
    Point the laser to a list of points in the world frame (mm) in order, using interpolation.
    
    Parameters:
    - cardNum: Laser card identifier.
    - points_mm: List of tuples (x_mm, y_mm) representing points in the world frame (mm).
    - mtx, dist: Camera calibration parameters.
    - picam2: Picamera2 instance.
    - marker_length_mm: Marker length in mm.
    - reference_id: Reference marker ID.
    - csv_path: Path to calibration CSV file.
    """
    mm_xy, bit_xy, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale = load_data(csv_path)
    
    # Go to initial position (0, 0) in laser reference frame
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.execute(cardNum)
    time.sleep(0.05)
    
    # Convert and point to each point
    for x_mm, y_mm in points_mm:
        x_bit, y_bit = project_to_bits(x_mm, y_mm, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)
        if x_bit is None or y_bit is None:
            print(f"Projection failed for ({x_mm}, {y_mm}). Stopping movement.")
            libe1701py.close(cardNum)
            return
        print(f"Pointing to ({x_mm}, {y_mm}) -> ({x_bit}, {y_bit})")

        pf.show_spot(mtx, dist, picam2, marker_length_mm, reference_id)

        libe1701py.jump_abs(cardNum, x_bit, y_bit, 0)
        libe1701py.execute(cardNum)
        lc.wait_marking(cardNum)
       
        time.sleep(0.5)  # Pause to observe each point

    # Return to origin
    libe1701py.execute(cardNum)
    time.sleep(0.2)
    lc.wait_marking(cardNum)
    time.sleep(0.1)
    libe1701py.close(cardNum)



    def go_to_pos(pos_mm, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale):
    """Go to a specific position in the world frame."""

    x_g, y_g = project_to_bits(*pos_mm, mm_xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)
    
    if x_g is not None and y_g is not None:
        print(f"Projected coordinates for {pos_mm}: ({x_g}, {y_g})")
        
        #cardNum = lc.init_laser()
        libe1701py.jump_abs(cardNum, 0, 0, 0)  # Initial position
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
        libe1701py.execute(cardNum)
        
        time.sleep(1)
        
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
        libe1701py.jump_abs(cardNum, x_g, y_g, 0)
        libe1701py.execute(cardNum)
        lc.wait_marking(cardNum)
        time.sleep(3)
        libe1701py.close(cardNum)
    else:
        print("Projection failed. Check the data and input.")