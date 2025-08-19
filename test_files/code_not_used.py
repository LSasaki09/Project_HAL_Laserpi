'''Part of code / functions not used '''



''' PICAMERA2 FUNCTIONS NOT USED IN THE MAIN SCRIPT '''
####################### picamFunctions #######################

''' Functions world for world frame (mm)'''


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















'''SCANNER FUNCTIONS NOT USED IN THE MAIN SCRIPT'''
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


#####################################################################################################################33

''' SCANNER FUNCTIONS NOT USED IN THE MAIN SCRIPT '''

def non_linear_func(x, *data):
    """
    x: coordinates to update
    - aruco_speed: speed of the aruco (supposed constant) in units of pixels/sec
    - marking_speed: speed of the marking in units of pixels/sec
    """
    start_pts, end_pts, aruco_speed, mark_speed = data
    t = np.linalg.norm(x-start_pts)/np.linalg.norm(mark_speed)
    eq1 = end_pts[0] + aruco_speed[0]*t
    eq2 = end_pts[1] + aruco_speed[1]*t
    return [eq1, eq2]


def create_moving_goal_pts(goal_pts, aruco_speed, marking_speed):
    """
    goal_pts: points of the pattern to draw in units of pixels
    - aruco_speed: speed of the aruco (supposed constant) in units of pixels/sec
    - marking_speed: speed of the marking in units of pixels/sec
    """
    for i in range(np.size(goal_pts, axis = 1))-1:
        sol = fsolve(non_linear_func, goal_pts[i+1], args = [goal_pts[i], goal_pts[i+1], aruco_speed, mark_speed])
        goal_pts[i+1] = sol
    return goal_pts


#######################################################################################################
def draw_square_world_frame(cardNum, square_size, x0, y0, xy_norm=None, bit_xy_norm=None, mm_min=None, mm_max=None, bit_scale=None):
    """
    Draw a square in the world frame (mm) by converting coordinates to laser frame (bits) using interpolation.
    
    Parameters:
    - cardNum: Laser card identifier.
    - square_size: Side length of the square in millimeters.
    - x0, y0: Center of the square in mm or pixels
    - xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale: Parameters for interpolation.
    """
    half_size = square_size / 2
    
    # Define the square corners in the world frame (mm)
    corners_mm = [
        (x0 - half_size, y0 - half_size),  # Bottom-left corner
        (x0 - half_size, y0 + half_size),  # Top-left corner
        (x0 + half_size, y0 + half_size),  # Top-right corner
        (x0 + half_size, y0 - half_size),  # Bottom-right corner
        (x0 - half_size, y0 - half_size)   # Back to bottom-left corner
    ]
    x_center, y_center = project_to_bits(x0, y0, xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)
    
    if x_center is None or y_center is None:
        print(f"Projection failed for the center ({x0}, {y0}).")
        return

    # Convert corners to laser coordinates (bits) using interpolation
    corners_bits = []
    for x_mm, y_mm in corners_mm:
        x_bit, y_bit = project_to_bits(x_mm, y_mm, xy_norm, bit_xy_norm, mm_min, mm_max, bit_scale)
        
        if x_bit is None or y_bit is None:
            print(f"Projection failed for ({x_mm}, {y_mm}). Stopping drawing.")
            return
        corners_bits.append((x_bit, y_bit))

    # Activate the laser
    libe1701py.jump_abs(cardNum, x_center, y_center, 0)  
    
    # Draw the square
    for x_bit, y_bit in corners_bits:
        libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)
    
    libe1701py.mark_abs(cardNum, x_center, y_center, 0)
    libe1701py.execute(cardNum)
    
    lc.wait_marking(cardNum)

#######################################################################################################

def draw_pattern(cardNum, square_size,pattern_name,x0, y0, xy_norm=None, bit_xy_norm=None, min_coord=None, max_coord=None, bit_scale=None):
    half_size = square_size / 2
    
    if pattern_name == "square":
        goal_pts = [
            (x0 - half_size, y0 - half_size),  # Bottom-left corner
            (x0 - half_size, y0 + half_size),  # Top-left corner
            (x0 + half_size, y0 + half_size),  # Top-right corner
            (x0 + half_size, y0 - half_size),  # Bottom-right corner
            (x0 - half_size, y0 - half_size)   # Back to bottom-left corner
        ]

    elif pattern_name == "spiral":
        n_turns = 6  # turn nb spiral
        points_per_turn = 20  # Points per turn 
        total_points = n_turns * points_per_turn
        goal_pts = []
        for i in range(total_points):
            angle = (i / points_per_turn) * 2 * np.pi
            radius = (i / total_points) * half_size
            x = x0 + radius * np.cos(angle)
            y = y0 + radius * np.sin(angle)
            goal_pts.append((x, y))


    elif pattern_name == "sandglass":
        goal_pts = [
            (x0 - half_size, y0 + half_size),  
            (x0 + half_size, y0 + half_size),  
            (x0 - half_size, y0 - half_size),                
    
            (x0 + half_size, y0 - half_size),
            (x0 - half_size, y0 + half_size)
        ]
    elif pattern_name == "zigzag":
        n_lines = 5  # nb horizontal line
        points_per_line = 2  # Points per line
        goal_pts = []
        for i in range(n_lines):
            y = y0 - half_size + (i / (n_lines - 1)) * square_size if n_lines > 1 else y0
            if i % 2 == 0:
                # leftto right
                for j in range(points_per_line):
                    x = x0 - half_size + (j / (points_per_line - 1)) * square_size
                    goal_pts.append((x, y))
            else:
                # right to left
                for j in range(points_per_line - 1, -1, -1):
                    x = x0 - half_size + (j / (points_per_line - 1)) * square_size
                    goal_pts.append((x, y))
    else:
        print(f"Unknown pattern: {pattern_name}. Supported patterns: square, spiral, sandglass, zigzag.")
        return
    x_center, y_center = project_to_bits(x0, y0, xy_norm, bit_xy_norm, min_coord,max_coord, bit_scale)
    
    if x_center is None or y_center is None:
        print(f"Projection failed for the center ({x0}, {y0}).")
        return

    # Convert corners to laser coordinates (bits) using interpolation
    goal_pts_bits = []
    for x_g, y_g in goal_pts:
        x_bit, y_bit = project_to_bits(x_g, y_g, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        
        if x_bit is None or y_bit is None:
            print(f"Projection failed for ({x_mm}, {y_mm}). Stopping drawing.")
            return
        goal_pts_bits.append((x_bit, y_bit))

    libe1701py.jump_abs(cardNum, x_center, y_center, 0)  
    
    # Draw 
    for x_bit, y_bit in goal_pts_bits:
        libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)
    
    #libe1701py.mark_abs(cardNum, x_center, y_center, 0)
    libe1701py.execute(cardNum)
    
    lc.wait_marking(cardNum)

##################################################################################################

def live_tracking_px(cardNum,picam2,track_id, csv_path="scanner_camera_map.csv"):

    unit = "pixels"  
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 1  # 
    x_center_old, y_center_old = 0., 0.  # Previous center coordinates

    # Go to initial position (0, 0) in laser reference frame
    #libe1701py.jump_abs(cardNum, 0, 0, 0)
    #libe1701py.execute(cardNum)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    
    libe1701py.execute(cardNum)

    while True:
        frame = picam2.capture_array()
        centers, ids,corners_list, frame = pf.detect_aruco_markers(frame,unit)

        small = cv2.resize(frame, (640, 480)) 
        cv2.imshow("Live Camera Feed", small)

        if track_id in ids:
            index = np.where(ids == track_id)[0][0]
            corners = corners_list[index].reshape(4, 2)
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            x_bit, y_bit = project_to_bits(center_x, center_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
            
            if x_bit is None or y_bit is None:
                print(f"Projection failed for center ({center_x:.2f}, {center_y:.2f}). Skipping.")
                continue
            
            #print(f"Tracking ID {track_id} at ({center_x:.2f}, {center_y:.2f}) → ({x_bit}, {y_bit})")

            if abs(center_x - x_center_old) < pixel_threshold and abs(center_y - y_center_old) < pixel_threshold:
                #print(f"ID {track_id} is stationary. Waiting for movement...")
                x_center_old, y_center_old = center_x, center_y
                libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                libe1701py.execute(cardNum)
                #draw_pattern_in_aruco(cardNum, "sandglass", corners, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
                continue

            #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.set_laser_delays(cardNum, 0.,50000.)  # Set laser delay to 5 ms
            libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)

            #libe1701py.jump_abs(cardNum, x_bit, y_bit, 0)
            #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            #draw_pattern_in_aruco(cardNum, "sandglass", corners, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)

            libe1701py.execute(cardNum)
            lc.wait_marking(cardNum)

            x_center_old, y_center_old = center_x, center_y

        else:
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
            print(f"ID {track_id} not detected. Waiting for detection...")

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

        time.sleep(0.01)

    libe1701py.close(cardNum)
    cv2.destroyAllWindows()


###############################################################################################

def go_to_several_points_without_cam(cardNum, points,shooting_time = 0.005, unit = "pixels", csv_path="scanner_camera_map.csv"):
    """
    Point the laser to a list of points in the world frame  using interpolation.
    
    Parameters:
    - cardNum: Laser card identifier.
    - points: List of tuples (x_coord, y_coord) representing points in the world frame (mm) or (pixels).
    - csv_path: Path to calibration CSV file.
    """

    coord_xy, bit_xy, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale = load_data(unit,csv_path)

    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    
    # Go to initial position (0, 0) in laser reference frame
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.execute(cardNum)
    time.sleep(0.1)
    
    # Convert and point to each point
    for x_coord, y_coord in points:
        x_bit, y_bit = project_to_bits(x_coord, y_coord, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale)

        if x_bit is None or y_bit is None:
            print(f"Échec de la projection pour ({x_coord}, {y_coord}). Arrêt du déplacement.")
            libe1701py.close(cardNum)
            return
        print(f"Pointing to ({x_coord}, {y_coord}) -> ({x_bit}, {y_bit})")

        libe1701py.jump_abs(cardNum, x_bit, y_bit, 0)
        #libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")

        #draw_pattern(cardNum,square_size, pattern_name,x_coord, y_coord, xy_norm, bit_xy_norm, pt_min, t_max, bit_scale)
        
        libe1701py.execute(cardNum)
        lc.wait_marking(cardNum)
       
        #time.sleep(0.1)

    libe1701py.close(cardNum)

############################################################################################################33




def live_multitracking_px(cardNum, picam2, track_id, track_vel_id=0, csv_path="scanner_camera_map.csv"):
    """
    Track multiple unique ArUco markers (all IDs != reference ID 0 are targets) for ~1 second each,
    using a reference marker (ID 0) to compute shared linear velocity. Prioritizes targets right-to-left
    if x-velocity > 0, left-to-right if <= 0. Maintains a memory of shot targets to avoid re-tracking.
    Start time for tracking begins when the laser starts tracking a target, not at detection.
    Note: track_id parameter is kept for compatibility but not used; all non-reference IDs are targets.
    """
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 2  # Threshold to detect movement
    camera_update_interval = 0.2  # Update camera every 0.2 seconds
    tracking_duration = 1.0  # Track each target for ~1 second
    shot_timeout = 3.0  # Remove shot targets not seen for 3 seconds

    # Initialize tracking variables
    last_center_track_vel_id = None
    shared_velocity = np.array([0.0, 0.0])  # Shared velocity from reference
    last_update_time = None
    last_camera_read_time = time.perf_counter() - camera_update_interval  # Force initial read
    active_targets = {}  # dict {id: {'center': np.array, 'time': float, 'start_time': float or None}}
    aruco_shooted = {}  # dict {id: {'center': np.array, 'time': float, 'shot_time': float}}

    # Turn on the laser at initial position
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)

    while True:
        current_time = time.perf_counter()

        # Check if it's time to update from camera (every 0.2s)
        if current_time - last_camera_read_time >= camera_update_interval:
            last_camera_read_time = current_time

            frame = picam2.capture_array()
            centers, ids, corners_list, frame = pf.detect_aruco_markers(frame, unit)

            small = cv2.resize(frame, (640, 480))
            cv2.imshow("Live Camera Feed", small)

            # Check for no detections
            if ids is None or len(ids) == 0:
                print(f"No markers detected during camera update. Continuing with last known velocity.")
                # Extrapolate shot targets' positions
                for target in aruco_shooted.values():
                    elapsed = current_time - target['time']
                    target['center'] += shared_velocity * elapsed
                    target['time'] = current_time
                # Remove shot targets not seen for too long
                aruco_shooted = {id_: t for id_, t in aruco_shooted.items() if current_time - t['time'] < shot_timeout}
                continue

            # Update shared velocity using reference marker (ID=0)
            if track_vel_id in ids:
                index_vel_id = np.where(ids == track_vel_id)[0][0]
                corners_vel_id = corners_list[index_vel_id].reshape(4, 2)
                current_center_vel = np.mean(corners_vel_id, axis=0)  # [x, y]

                if last_center_track_vel_id is None:
                    last_center_track_vel_id = current_center_vel
                    last_update_time = current_time
                    print(f"Initial detection of reference ID {track_vel_id} at ({current_center_vel[0]:.2f}, {current_center_vel[1]:.2f})")
                else:
                    dt = current_time - last_update_time
                    if dt > 1e-6:
                        shared_velocity = (current_center_vel - last_center_track_vel_id) / dt
                    else:
                        shared_velocity = np.array([0.0, 0.0])
                    print(f"Updated shared velocity from reference: ({shared_velocity[0]:.2f}, {shared_velocity[1]:.2f})")
                last_center_track_vel_id = current_center_vel
                last_update_time = current_time
            else:
                print(f"Reference ID {track_vel_id} not detected. Using last velocity.")

            '''
            # Check if stationary
            if np.linalg.norm(shared_velocity) * camera_update_interval < pixel_threshold:
                print("All targets are stationary based on reference velocity.")
                libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                libe1701py.execute(cardNum)
                active_targets.clear()
                aruco_shooted.clear()
                continue
            '''

            # Process all detected markers (targets are all IDs != track_vel_id)
            for i, marker_id in enumerate(ids):
                marker_id = marker_id.item()  # Convert NumPy array to scalar integer
                if marker_id != track_vel_id:
                    corners = corners_list[i].reshape(4, 2)
                    center = np.mean(corners, axis=0)
                    # Update or add to active_targets if not shot
                    if marker_id not in aruco_shooted:
                        if marker_id in active_targets:
                            active_targets[marker_id]['center'] = center
                            active_targets[marker_id]['time'] = current_time
                            print(f"Updated active target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")
                        else:
                            active_targets[marker_id] = {
                                'center': center,
                                'time': current_time,
                                'start_time': None  # Start time set when tracking begins
                            }
                            print(f"Added new target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")
                    else:
                        # Update shot target position
                        aruco_shooted[marker_id]['center'] = center
                        aruco_shooted[marker_id]['time'] = current_time
                        #print(f"Updated shot target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")

            '''
            # Extrapolate positions for shot targets not detected
            for marker_id in list(aruco_shooted):
                if marker_id not in [mid.item() for mid in ids]:
                    elapsed = current_time - aruco_shooted[marker_id]['time']
                    aruco_shooted[marker_id]['center'] += shared_velocity * elapsed
                    aruco_shooted[marker_id]['time'] = current_time
                if current_time - aruco_shooted[marker_id]['time'] > shot_timeout:
                    del aruco_shooted[marker_id]
                    print(f"Removed shot target ID {marker_id} due to timeout.")
            '''

            # Extrapolate positions for active targets not detected
            for marker_id in list(active_targets):
                if marker_id not in [mid.item() for mid in ids]:
                    elapsed = current_time - active_targets[marker_id]['time']
                    active_targets[marker_id]['center'] += shared_velocity * elapsed
                    active_targets[marker_id]['time'] = current_time

        # Scanner update: track the prioritized target if any
        if active_targets:
            # Sort active IDs by x-coordinate based on velocity direction
            active_ids = list(active_targets.keys())
            if shared_velocity[0] > 0:
                active_ids.sort(key=lambda mid: active_targets[mid]['center'][0], reverse=False)  # Right to left
                #print("Prioritizing targets right-to-left (positive x-velocity).")
            else:
                active_ids.sort(key=lambda mid: active_targets[mid]['center'][0])  # Left to right
                #print("Prioritizing targets left-to-right (negative or zero x-velocity).")

            current_id = active_ids[0]
            current_target = active_targets[current_id]

            # Set start_time when tracking begins
            if current_target['start_time'] is None:
                current_target['start_time'] = current_time
                print(f"Started tracking target ID {current_id} at ({current_target['center'][0]:.2f}, {current_target['center'][1]:.2f})")

            # Check if tracking duration exceeded
            if current_time - current_target['start_time'] > tracking_duration:
                print(f"Finished tracking target ID {current_id} at ({current_target['center'][0]:.2f}, {current_target['center'][1]:.2f}). Moving to shot list.")
                aruco_shooted[current_id] = {
                    'center': current_target['center'],
                    'time': current_time,
                    'shot_time': current_time
                }
                del active_targets[current_id]
                continue

            # Extrapolate current target's position
            elapsed = current_time - current_target['time']
            extrapolated_center = current_target['center'] + shared_velocity * elapsed
            extrapolated_x, extrapolated_y = extrapolated_center

            # Project to bits
            extrapolated_x_bit, extrapolated_y_bit = project_to_bits(
                extrapolated_x, extrapolated_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale
            )

            if extrapolated_x_bit is None or extrapolated_y_bit is None:
                print(f"Projection failed for extrapolated center ({extrapolated_x:.2f}, {extrapolated_y:.2f}). Skipping.")
                continue

            # Move scanner
            libe1701py.jump_abs(cardNum, extrapolated_x_bit, extrapolated_y_bit, 0)
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
        else:
            # No targets: Keep laser on but no move
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
            #print("No active targets to track.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep for ~0.0001s for fast scanner updates
        time.sleep(0.00001)

    libe1701py.close(cardNum)
    cv2.destroyAllWindows()

    ###############################################################################################