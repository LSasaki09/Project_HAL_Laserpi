import cv2
import numpy as np
import time
import csv
from picamera2 import Picamera2
import libe1701py
from cv2 import aruco
import picamFunctions as pf
import laserControl as lc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from multiprocessing import Process, Value, Event, Array, Lock


def close_all_devices(cardNum, picam2):

    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.close(cardNum)
    picam2.stop()
    cv2.destroyAllWindows()
    print("Laser off. Camera closed.")

def load_data(unit = "pixels",csv_path="scanner_camera_map.csv", SPOT_INTENSITY_TRESHOLD=220):
    """
    Loads data from a CSV file, removes invalid initial lines, 
    eliminates outliers using intensity threshold, then normalizes the cleaned data.
    
    Parameters:
    - csv_path: Path to the CSV file.
    - SPOT_INTENSITY_TRESHOLD: Threshold for spot intensity to filter outliers.
    
    Returns:
    - pts_xy: Cleaned (x_mm, y_mm) or (x_px, y_px) coordinates. 
    - bit_xy: Cleaned (bit_x, bit_y) coordinates.
    - xy_norm: Normalized (x_mm, y_mm) or (x_px, y_px) coordinates.
    - bit_xy_norm: Normalized (bit_x, bit_y) coordinates.
    - pt_min: Minimum values of (x_mm, y_mm) or (x_px, y_px).
    - pt_max: Maximum values of (x_mm, y_mm) or (x_px, y_px).
    - bit_scale: Scale factor for (bit_x, bit_y) coordinates.
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    #spot_intensity = df['intensity_pixel'].values
    spot_detected = df['spot_detected'].values
    
    # Create a mask to identify rows where intensity is above the threshold
    #mask = spot_intensity > SPOT_INTENSITY_TRESHOLD
    mask = spot_detected == True
    # Filter the DataFrame to remove outliers
    df_clean = df[mask]
    
    # Display the number of outliers removed
    num_outliers = len(df) - len(df_clean)
    print(f"Removing {num_outliers} outliers.")
    
    # Extract cleaned data
    if unit == "mm":
        bit_xy = df_clean[['bit_x', 'bit_y']].values
        pts_xy = df_clean[['x_mm', 'y_mm']].values 
    elif unit == "pixels":
        bit_xy = df_clean[['bit_x', 'bit_y']].values
        pts_xy = df_clean[['x_px', 'y_px']].values
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        return None, None, None, None, None, None, None
    
    # Normalization
    bit_scale = 1e7
    bit_xy_norm = bit_xy / bit_scale
    
    pt_min = np.min(pts_xy, axis=0)
    pt_max = np.max(pts_xy, axis=0)
    xy_norm = (pts_xy - pt_min) / (pt_max - pt_min)
    
    return pts_xy, bit_xy, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale

def project_to_bits(x, y, xy_norm, bit_xy_norm, xy_min, xy_max, bit_scale):
    # Normalize inputs
    x_norm = (x - xy_min[0]) / (xy_max[0] - xy_min[0])
    y_norm = (y - xy_min[1]) / (xy_max[1] - xy_min[1])

    # Check if normalized values are within [0, 1]
    if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
        print("Warning: (x, y) coordinates are outside the data range. The result may be inaccurate.")

    # Interpolate using normalized data
    bit_x_norm = griddata(xy_norm, bit_xy_norm[:, 0], (x_norm, y_norm), method='cubic')
    bit_y_norm = griddata(xy_norm, bit_xy_norm[:, 1], (x_norm, y_norm), method='cubic')

    # Handle NaN results
    if np.isnan(bit_x_norm) or np.isnan(bit_y_norm):
        print("Error: Interpolation failed. Check the data range.")
        return None, None

    # Denormalize outputs
    bit_x = bit_x_norm * bit_scale
    bit_y = bit_y_norm * bit_scale

    return int(bit_x), int(bit_y)

def draw_pattern_in_aruco(cardNum, pattern_name, corners, xy_norm=None, bit_xy_norm=None, min_coord=None, max_coord=None, bit_scale=None, aruco_speed=0, marking_speed = 0):
    """
    Draw a pattern within the boundaries defined by an ArUco marker's corners, respecting its orientation.

    Parameters:
    - cardNum: Laser card identifier.
    - pattern_name: Name of the pattern to draw ('square', 'spiral', 'sandglass', 'zigzag').
    - corners: Numpy array of shape (4, 2) containing the four corners of the ArUco marker.
    - xy_norm: Normalized calibration coordinates (mm or pixels).
    - bit_xy_norm: Normalized bit coordinates.
    - min_coord, max_coord: Min/max coordinates for interpolation.
    - bit_scale: Scaling factor for bit coordinates.
    - aruco_speed: speed of the aruco (supposed constant) in units of pixels/sec
    - marking_speed: speed of the marking in units of pixels/sec
    """
    # Ensure corners are in the correct shape
    corners = corners.reshape(4, 2)

    # Define a unit square for homography mapping (in normalized [0,1]x[0,1] space)
    unit_square = np.array([
        [0,1],                        #[0, 0],  # Bottom-left
        [1,1],                        #[0, 1],  # Top-left
        [1,0],                        #[1, 1],  # Top-right
        [0,0]                        #[1, 0]   # Bottom-right
    ], dtype=np.float32)

    # Calculate homography from unit square to ArUco marker corners
    h_matrix, _ = cv2.findHomography(unit_square, corners, cv2.RANSAC)

    # Calculate center for initial laser jump
    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])

    if pattern_name == "square":
        # Use the exact corners of the ArUco marker
        goal_pts = corners.tolist() + [corners[0].tolist()]  # Close the loop

    elif pattern_name == "spiral":
        n_turns = 4  # Number of spiral turns
        points_per_turn = 10  # Points per turn
        total_points = n_turns * points_per_turn
        goal_pts_unit = []
        for i in range(total_points):
            angle = (i / points_per_turn) * 2 * np.pi
            radius = (i / total_points) * 0.5  # Max radius is 0.5 to stay within unit square
            x = 0.5 + radius * np.cos(angle)  # Center at (0.5, 0.5)
            y = 0.5 + radius * np.sin(angle)
            goal_pts_unit.append([x, y])
        # Transform points to marker space
        goal_pts_unit = np.array(goal_pts_unit, dtype=np.float32).reshape(-1, 1, 2)
        goal_pts = cv2.perspectiveTransform(goal_pts_unit, h_matrix).reshape(-1, 2).tolist()

    elif pattern_name == "several_spirals":
        n_turns = 4  # Number of spiral turns
        points_per_turn = 18  # Points per turn
        total_points = n_turns * points_per_turn
        goal_pts_unit = []
        for j in range (10):
            for i in range(total_points):
                angle = (i / points_per_turn) * 2 * np.pi
                radius = (i / total_points) * 0.5  # Max radius is 0.5 to stay within unit square
                x = 0.5 + radius * np.cos(angle)  # Center at (0.5, 0.5)
                y = 0.5 + radius * np.sin(angle)
                goal_pts_unit.append([x, y])
        # Transform points to marker space
        goal_pts_unit = np.array(goal_pts_unit, dtype=np.float32).reshape(-1, 1, 2)
        goal_pts = cv2.perspectiveTransform(goal_pts_unit, h_matrix).reshape(-1, 2).tolist()


    elif pattern_name == "sandglass":
        # Use the exact corners in sandglass order
        goal_pts = [
            corners[1].tolist(),  # Top-left
            corners[2].tolist(),  # Top-right
            corners[0].tolist(),  # Bottom-left
            corners[3].tolist(),  # Bottom-right
            corners[1].tolist()   # Back to top-left
        ]

    elif pattern_name == "several_sandglasses":
        goal_pts = []
        base_pattern = [
            corners[1].tolist(),  # Top-left
            corners[2].tolist(),  # Top-right
            corners[0].tolist(),  # Bottom-left
            corners[3].tolist(),  # Bottom-right
            corners[1].tolist()   # Back to top-left
        ]
        for _ in range(10):
            goal_pts.extend(base_pattern)
    

    elif pattern_name == "zigzag":
        n_lines = 5  # Number of diagonal lines
        goal_pts_unit = []
        for i in range(n_lines):
            t = i / (n_lines - 1) if n_lines > 1 else 0.5
            if i % 2 == 0:
                # Bottom-left to top-right diagonal
                start_x, start_y = 0, t
                end_x, end_y = 1, 1 - t
            else:
                # Bottom-right to top-left diagonal
                start_x, start_y = 1, t
                end_x, end_y = 0, 1 - t
            goal_pts_unit.append([start_x, start_y])
            goal_pts_unit.append([end_x, end_y])
        goal_pts_unit.append([0, 0])  # Return to start for continuity
        # Transform points to marker space
        goal_pts_unit = np.array(goal_pts_unit, dtype=np.float32).reshape(-1, 1, 2)
        goal_pts = cv2.perspectiveTransform(goal_pts_unit, h_matrix).reshape(-1, 2).tolist()

    elif pattern_name == "simple_zigzag":
        x, y = corners[1].tolist()[0], corners[1].tolist()[1]
        x1, y1 = project_to_bits(x, y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        x, y = corners[2].tolist()[0], corners[2].tolist()[1]
        x2, y2 = project_to_bits(x, y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        #if y1 is None or y2 is None:
        #    Max_wobble_amplitude = 10000
        #else:
        Max_wobble_amplitude = int(np.absolute(y1 - y2) * 0.5)
        libe1701py.set_wobble(cardNum, Max_wobble_amplitude//1000, Max_wobble_amplitude, 250)
        #libe1701py.execute(cardNum)
        goal_pts_unit = [[0, 0.5], [1, 0.5]]
        goal_pts_unit = np.array(goal_pts_unit, dtype=np.float32).reshape(-1, 1, 2)
        goal_pts = cv2.perspectiveTransform(goal_pts_unit, h_matrix).reshape(-1, 2).tolist()

    else:
        print(f"Unknown pattern: {pattern_name}. Supported patterns: square, spiral, sandglass, zigzag.")
        return

    # Convert center to bits
    x_center, y_center = project_to_bits(center_x, center_y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
    if x_center is None or y_center is None:
        print(f"Projection failed for the center ({center_x}, {center_y}).")
        return

    # Convert pattern points to laser coordinates (bits)
    goal_pts_bits = []
    for x_g, y_g in goal_pts:
        x_bit, y_bit = project_to_bits(x_g, y_g, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        if x_bit is None or y_bit is None:
            print(f"Projection failed for ({x_g}, {y_g}). Stopping drawing.")
            return
        goal_pts_bits.append((x_bit, y_bit))

    #libe1701py.jump_abs(cardNum, x_center, y_center, 0)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")

    # Draw the pattern

    #Max_wobble_amplitude = 10000000
    for x_bit, y_bit in goal_pts_bits:
        #libe1701py.set_wobble(cardNum, Max_wobble_amplitude//15, Max_wobble_amplitude//15, 1000)
        libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)

    #libe1701py.execute(cardNum)
    #lc.wait_marking(cardNum)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")

def draw_in_aruco_during_time(cardNum, center_x,center_y,pattern_name, corners,shoot_time = 0.0025,  xy_norm=None, bit_xy_norm=None, min_coord=None, max_coord=None, bit_scale=None, aruco_speed=0, marking_speed = 0):
    """
    Draw repeatidly a pattern in aruco for X = shoot_time seconds.
    """
    x_bit, y_bit = project_to_bits(center_x, center_y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
    libe1701py.jump_abs(cardNum,x_bit , y_bit, 0)  # Jump to initial position
    libe1701py.execute(cardNum)
    time.sleep(0.0025)  # Allow time for the jump to complete
    start_time = time.perf_counter()
    while True:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > shoot_time:
            break
        
        # Draw the pattern within the ArUco marker
        draw_pattern_in_aruco(cardNum, pattern_name, corners, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale, aruco_speed, marking_speed)
        libe1701py.execute(cardNum)
        lc.wait_marking(cardNum)

def shoot_target_by_priority_px(cardNum, picam2, csv_path="scanner_camera_map.csv"):
    """
    Will shoot targets (aruco) by order or size.
    Will also shoot bigger ones for longer.
    """
    # Load calibration mapping for pixel → scanner coordinates
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data("pixels", csv_path)

    # Capture frame and detect markers
    frame = picam2.capture_array()
    centers, ids, corners_list, frame = pf.detect_aruco_markers(frame, "pixels", 0, 0)

    if ids is None or len(ids) == 0:
        print("No ArUco markers detected.")
        return

    # Store markers with their sizes and positions
    aruco_info = []
    for idx, corner in enumerate(corners_list):
        if corner is not None:
            corner = corner.reshape(4, 2)
            # Compute all four side lengths
            side_lengths = [
                np.linalg.norm(corner[1] - corner[0]),
                np.linalg.norm(corner[2] - corner[1]),
                np.linalg.norm(corner[3] - corner[2]),
                np.linalg.norm(corner[0] - corner[3])
            ]
            avg_size = np.mean(side_lengths)  # Average side length in pixels
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])
            aruco_info.append({
                "id": ids[idx][0],
                "size": avg_size,
                "center": (center_x, center_y),
                "corners": corner
            })

    # Sort by size (largest first)
    #aruco_info.sort(key=lambda x: x["size"], reverse=True)
    aruco_info.sort(key=lambda x: x["center"][0], reverse=False)

    # Define thresholds for categories (you can tune these)
    BIG_THRESHOLD = 80 #170
    MID_THRESHOLD = 50 #120
    coeff_time = 0.3 # 0.01 for ms units 0.3

    for aruco in aruco_info:
        size = aruco["size"]
        center_x, center_y = aruco["center"]

        # Categorize by size
        if size >= BIG_THRESHOLD:
            category = "big"
            shoot_time = 2.5 * coeff_time   # 25 ms
        elif size >= MID_THRESHOLD:
            category = "mid"
            shoot_time = 1.2 * coeff_time
        else:
            category = "little"
            shoot_time = 0.6 * coeff_time

        print(f"Aruco ID {aruco['id']} → size: {size:.2f}px → {category} target")


        # Project pixel coordinates to scanner coordinates
        x_bit, y_bit = project_to_bits(center_x, center_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
        if x_bit is None or y_bit is None:
            print("Projection failed, skipping target.")
            continue

        
        corners = aruco["corners"]
        
        # Move laser to target and fire
        #lc.punching(cardNum, [x_bit,y_bit], shoot_time, True)
        draw_in_aruco_during_time(cardNum, center_x,center_y,"sandglass", corners,shoot_time,  coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)

        #libe1701py.execute(cardNum)
        #lc.wait_marking(cardNum)

def go_to_aruco(cardNum, corners_list, unit="mm", pattern_name="square", csv_path="scanner_camera_map.csv"):
    """
    Draw patterns within the boundaries of detected ArUco markers.

    Parameters:
    - cardNum: Laser card identifier.
    - corners_list: List of corner arrays for detected ArUco markers.
    - unit: 'mm' or 'pixels' for coordinate system.
    - pattern_name: Pattern to draw ('square', 'spiral', 'sandglass', 'zigzag').
    - csv_path: Path to calibration CSV file.
    """
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)

    # Go to initial position (0, 0) in laser reference frame
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.execute(cardNum)
    #time.sleep(0.1)

    # Draw pattern for each marker's corners
    for corners in corners_list:
        corners = corners.reshape(4, 2)
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        x_bit, y_bit = project_to_bits(center_x, center_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
        if x_bit is None or y_bit is None:
            print(f"Projection failed for center ({center_x:.2f}, {center_y:.2f}). Skipping.")
            continue
        print(f"Drawing {pattern_name} at center ({center_x:.2f}, {center_y:.2f})")

        draw_pattern_in_aruco(cardNum, pattern_name, corners, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
    
    libe1701py.execute(cardNum)
    lc.wait_marking(cardNum)

    time.sleep(0.1)
    libe1701py.close(cardNum)

def live_tracking_px_predict(cardNum, picam2, track_id, csv_path="scanner_camera_map.csv"):
    """
    Closed loop tracking of the Aruco position
    """
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 2  # Threshold to detect movement
    delay = 0.1  # Estimated total delay in seconds

    # Initialize position history
    prev_center = None
    prev_time = None

    # Turn on the laser
    libe1701py.jump_abs(cardNum, 0, 0, 0)  # Go to initial position (0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)
    
    while True:
        frame = picam2.capture_array()
        centers, ids, corners_list, frame = pf.detect_aruco_markers(frame, unit)

        small = cv2.resize(frame, (640, 480))
        cv2.imshow("Live Camera Feed", small)

        if track_id in ids:
            index = np.where(ids == track_id)[0][0]
            corners = corners_list[index].reshape(4, 2)
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            current_center = np.array([center_x, center_y])

            # Project current position to bits
            x_bit, y_bit = project_to_bits(center_x, center_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)

            if x_bit is None or y_bit is None:
                print(f"Projection failed for center ({center_x:.2f}, {center_y:.2f}). Skipping.")
                continue

            current_time = time.perf_counter()
            # First detection: initialize previous position and time
            if prev_center is None:
                prev_center = current_center
                prev_time = current_time
                continue

            # Calculate velocity (pixels per second)
            dt = current_time - prev_time
            velocity = (current_center - prev_center) / dt if dt > 0 else np.array([0, 0])
            prev_time = time.perf_counter()

            # Predict future position
            future_center = current_center + velocity* delay 
            future_x, future_y = future_center

            # Project future position to bits
            future_x_bit, future_y_bit = project_to_bits(future_x, future_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)

            if future_x_bit is None or future_y_bit is None:
                print(f"Projection failed for future center ({future_x:.2f}, {future_y:.2f}). Skipping.")
                continue

            print(f"Tracking ID {track_id} at ({center_x:.2f}, {center_y:.2f}) → Predicted ({future_x:.2f}, {future_y:.2f}) → ({future_x_bit}, {future_y_bit})")

            if abs(current_center[0] - prev_center[0]) < pixel_threshold and abs(current_center[1] - prev_center[1]) < pixel_threshold:
                #print(f"ID {track_id} is stationary. Waiting for movement...")
                libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                libe1701py.execute(cardNum)
                #draw_pattern_in_aruco(cardNum, "sandglass", corners, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
                continue
            
            # Move to predicted position
            libe1701py.mark_abs(cardNum, future_x_bit, future_y_bit, 0)
            #libe1701py.jump_abs(cardNum, future_x_bit, future_y_bit, 0)
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
            #lc.wait_marking(cardNum)

            # Update previous position
            prev_center = current_center
            #prev_time = current_time

        else:
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
            print(f"ID {track_id} not detected. Waiting for detection...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    libe1701py.close(cardNum)
    cv2.destroyAllWindows()

def live_tracking_px_v2(cardNum, picam2, track_id,track_vel_id=0, csv_path="scanner_camera_map.csv"):
    """
    Closed loop but Camera updates velocity and position at delayed intervals.
    """
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 2  # Threshold to detect movement
    camera_update_interval = 0.2  # Update camera every X seconds

    # Initialize tracking variables
    last_center_track_vel_id = None
    last_center_track_id = None
    last_velocity = np.array([0.0, 0.0])  # Initial velocity
    last_update_time = None
    last_camera_read_time = time.perf_counter() - camera_update_interval  # Force initial read

    # Turn on the laser at initial position
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)

    while True:
        current_time = time.perf_counter()

        # Check if it's time to update from camera (every x seconds)
        if current_time - last_camera_read_time >= camera_update_interval:

            #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")  # Turn off laser for camera read
            #libe1701py.execute(cardNum)
            #time.sleep(1/40)
            last_camera_read_time = current_time

            frame = picam2.capture_array()
            #time.sleep(0.01)
            centers, ids, corners_list, frame = pf.detect_aruco_markers(frame, unit)

            small = cv2.resize(frame, (640, 480))
            cv2.imshow("Live Camera Feed", small)

            # Check for no detections early
            if ids is None or len(ids) == 0:
                print(f"No markers detected during camera update. Continuing with last known velocity.")
                continue
            
            if track_vel_id in ids:
                index_vel_id = np.where(ids == track_vel_id)[0][0]
                corners_vel_id = corners_list[index_vel_id].reshape(4, 2)
                current_center_track_vel_id = np.mean(corners_vel_id, axis=0)  # [x, y]

                # If first detection, initialize
                if last_center_track_vel_id is None:
                    last_center_track_vel_id = current_center_track_vel_id
                    last_update_time = current_time
                    print(f"Initial detection of ID {track_vel_id} at ({current_center_track_vel_id[0]:.2f}, {current_center_track_vel_id[1]:.2f})")
                    continue

                # Calculate dt and velocity
                dt = current_time - last_update_time
                if dt > 0:
                    last_velocity = (current_center_track_vel_id - last_center_track_vel_id) / dt
                else:
                    last_velocity = np.array([0.0, 0.0])
                

            if track_id in ids:
                index_track_id = np.where(ids == track_id)[0][0]
                corners_track_id = corners_list[index_track_id].reshape(4, 2)
                current_center_track_id = np.mean(corners_track_id, axis=0)  # [x, y]

                if last_center_track_id is None:
                    last_center_track_id = current_center_track_id
                    #last_update_time = current_time
                    print(f"Initial detection of ID {track_id} at ({current_center_track_id[0]:.2f}, {current_center_track_id[1]:.2f})")
                    continue


                '''
                 # Check if stationary
                if abs(current_center_track_vel_id[0] - last_center_track_vel_id[0]) < pixel_threshold and abs(current_center_track_vel_id[1] - last_center_track_vel_id[1]) < pixel_threshold:
                    # Stationary: No need to move, but keep laser on
                    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                    libe1701py.execute(cardNum)
                    continue
                '''

                # Check if stationary (using velocity magnitude)
                if np.linalg.norm(last_velocity) * camera_update_interval < pixel_threshold:
                    # Stationary: No need to move, but keep laser on
                    print(f"ID {track_id} is stationary.")
                    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                    libe1701py.execute(cardNum)
                    continue
                

                # Update last known position and time
                last_center_track_vel_id = current_center_track_vel_id
                last_center_track_id = current_center_track_id
                last_update_time = current_time

                print(f"Camera update: ID {track_id} at ({current_center_track_id[0]:.2f}, {current_center_track_id[1]:.2f}), velocity ({last_velocity[0]:.2f}, {last_velocity[1]:.2f})")

            else:
                # Target lost: Keep using last velocity/position for extrapolation, or handle as needed
                print(f"ID {track_id} not detected during camera update")
                # Optionally reset velocity to 0 if lost for too long, but for now continue

        # Extrapolate position for scanner update (every loop ~0.0001s, but in practice faster)
        if last_center_track_id is not None and last_update_time is not None:
            elapsed = current_time - last_update_time
            extrapolated_center = last_center_track_id + last_velocity * elapsed
            extrapolated_x, extrapolated_y = extrapolated_center

            # Project extrapolated position to bits
            extrapolated_x_bit, extrapolated_y_bit = project_to_bits(
                extrapolated_x, extrapolated_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale
            )

            if extrapolated_x_bit is None or extrapolated_y_bit is None:
                print(f"Projection failed for extrapolated center ({extrapolated_x:.2f}, {extrapolated_y:.2f}). Skipping.")
                continue

            # Move scanner to extrapolated position
            #libe1701py.mark_abs(cardNum, extrapolated_x_bit, extrapolated_y_bit, 0)
            libe1701py.jump_abs(cardNum, extrapolated_x_bit, extrapolated_y_bit, 0)
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)

            #print(f"Scanner update: Extrapolated to ({extrapolated_x:.2f}, {extrapolated_y:.2f}) → ({extrapolated_x_bit}, {extrapolated_y_bit})")
        else:
            # No target yet: Keep laser on but no move
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep for ~0.0001s to approximate fast scanner updates (adjust as needed for CPU)
        time.sleep(0.00001)

    libe1701py.close(cardNum)
    cv2.destroyAllWindows()

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
    #shot_timeout = 3.0  # Remove shot targets not seen for 3 seconds

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

            '''
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
            '''

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
            ###############################################################################33
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
                            #print(f"Updated active target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")
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
            #print(f"elapsed time {elapsed:.3f}s")
            print(f"current time {current_time:.3f}s, target time {current_target['time']:.3f}s")
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

def targets_detection(last_camera_read_time, last_center_track_vel_id, shared_velocity, track_vel_id,
                        ids, corners_list, centers, lock, unit, MAX_TARGETS,stop_event):
    """
    Detection process: Updates centers, corners, IDs, and velocity in shared memory.
    """
    camera_update_interval = 0.1 #0.1  # Update camera every 0.2 seconds
    picam2, mtx, dist = pf.init_camera() #ExposureTime = 40000, gain = 2.4

    while True:
        current_time = time.perf_counter()

        # Check if it's time to update from camera (every 0.2s)

        #frame = picam2.capture_array()

        request = picam2.capture_request()
        metadata = request.get_metadata() # metadata dict
        delay = metadata["SensorTimestamp"] * 1e-9 - current_time
        frame = request.make_array("main") # your image
        request.release()
        
        local_centers, local_ids, local_corners_list, frame = pf.detect_aruco_markers(frame, unit)
        #print(f"delay equals : {delay} seconds")
        
        with lock:
            last_camera_read_time.value = current_time - delay
            """
            Substracting delay to last_camera_read_time makes it wrong in theroy.
            It is done such that the picam2.capture_array() is taken into account.
                lag : arr = picam2.capture_array() waits until the next frame and registers it into arr. The total lag is thus lag = waiting time until next frame (wtnf) + process time.
                Whitout accounting for delay (wich equals wtnf), the informations of positions sent are already late by wtnf seconds. Accounting for delay could be done by already
                computing the new center swifted by delay * velocity. However, this should be equivalent to just shifting the time reference by that same amount.
            """
        #print (f"LAST CAMERA READ VALUE: {last_camera_read_time.value}")       

        #small = cv2.resize(frame, (640, 480))
        #cv2.imshow("Live Camera Feed", small)

        # Check for no detections
        if local_ids is None or len(local_ids) == 0:
            print(f"No markers detected during camera update. Continuing with last known velocity.")
            continue

        with lock:
            # Update shared velocity using reference marker (ID=0)
            if track_vel_id.value in local_ids:
                index_vel_id = np.where(local_ids == track_vel_id.value)[0][0]
                corners_vel_id = local_corners_list[index_vel_id].reshape(4, 2)
                current_center_vel = np.mean(corners_vel_id, axis=0)  # [x, y]

                if all(v == 0 for v in last_center_track_vel_id):
                    last_center_track_vel_id[0] = current_center_vel[0]
                    last_center_track_vel_id[1] = current_center_vel[1]
                    last_update_time = current_time  # Local, as it's not shared
                    print(f"Initial detection of reference ID {track_vel_id.value} at ({current_center_vel[0]:.2f}, {current_center_vel[1]:.2f})")
                else:
                    dt = current_time - last_update_time
                    print(f"TIME update camera: {current_time:.3f} seconds")

                    if dt > 1e-6:
                        shared_velocity[0] = (current_center_vel[0] - last_center_track_vel_id[0]) / dt
                        shared_velocity[1] = (current_center_vel[1] - last_center_track_vel_id[1]) / dt
                    else:
                        shared_velocity[0] = 0.0
                        shared_velocity[1] = 0.0
                    print(f"Updated shared velocity from reference: ({shared_velocity[0]:.2f}, {shared_velocity[1]:.2f})")
                last_center_track_vel_id[0] = current_center_vel[0]
                last_center_track_vel_id[1] = current_center_vel[1]
                last_update_time = current_time
            else:
                print(f"Reference ID {track_vel_id.value} not detected. Using last velocity.")

            # Update shared ids, centers, corners_list
            num_ids = min(len(local_ids), MAX_TARGETS)
            for i in range(num_ids):
                ids[i] = local_ids[i].item()  # Convert to scalar
                centers[2*i] = local_centers[i][0]
                centers[2*i + 1] = local_centers[i][1]
                flat_corners = local_corners_list[i].reshape(-1)
                for j in range(8):  # 4 corners * 2 (x,y)
                    corners_list[8*i + j] = flat_corners[j]
            # Clear remaining if fewer detected
            for i in range(num_ids, MAX_TARGETS):
                ids[i] = -1  # Invalid ID

        
        if stop_event.is_set():
            print("Stopping detection process.")
            cv2.destroyAllWindows()
            break

        # Sleep a bit to avoid busy loop in detection process
        time_end_process = time.perf_counter()
        process_time = time_end_process - current_time
        sleep_time = camera_update_interval - process_time

        if sleep_time<0 :
            sleep_time = 0.000001

        time.sleep(sleep_time)
        
def live_multitracking_multiprocess_px(cardNum, track_vel_id=0, csv_path="scanner_camera_map.csv"):
    """
    Track multiple unique ArUco markers (all IDs != reference ID 0 are targets) for ~1 second each,
    using a reference marker (ID 0) to compute shared linear velocity. Prioritizes targets right-to-left
    if x-velocity > 0, left-to-right if <= 0. Maintains a memory of shot targets to avoid re-tracking.
    Start time for tracking begins when the laser starts tracking a target, not at detection.
    Note: track_id parameter is kept for compatibility but not used; all non-reference IDs are targets.
    """
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 3  # Threshold to detect movement
    tracking_duration = 0.1 # Track each target for ~x second
    #shot_timeout = 3.0  # Remove shot targets not seen for 3 seconds
    MAX_TARGETS = 100  # Maximum number of ArUco markers expected

    Start_counting_for_end = time.perf_counter() 

    stop_event = Event()

    # Shared memory setup
    last_camera_read_time = Value('d', time.perf_counter())
    last_center_track_vel_id = Array('d', [0.0, 0.0])
    shared_velocity = Array('d', [0.0, 0.0])  # Shared velocity from reference
    t_vel_id = Value('i', track_vel_id)  # Track ID for reference marker
    shared_ids = Array('i',  [-1]*MAX_TARGETS)  # Shared array for detected IDs (-1 invalid)
    shared_corners_list = Array('d', [0.0] * (8 * MAX_TARGETS))  # 4 corners * 2 (x,y) per marker
    shared_centers = Array('d', [0.0] * (2 * MAX_TARGETS))  # x,y per marker
    #print(f"shared_ids: {shared_ids[:10]}")  # Debugging output
    lock = Lock()  # Lock for shared access

    # Start detection process
    cam_process = Process(target=targets_detection, args=(last_camera_read_time, last_center_track_vel_id, shared_velocity, 
                            t_vel_id, shared_ids, shared_corners_list, shared_centers, lock, unit, MAX_TARGETS, stop_event))
    cam_process.start()

    # Initialize main tracking variables
    active_targets = {}  # dict {id: {'center': np.array, 'time': float, 'start_time': float or None}}
    aruco_shooted = {}  # dict {id: {'center': np.array, 'time': float, 'shot_time': float}}

    change_target = True
    tracking_on = True
    # Turn on the laser at initial position
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)
    try:
        while True:
            current_time = time.perf_counter()

            # Read from shared memory
            with lock:
                local_ids = [shared_ids[i] for i in range(MAX_TARGETS) if shared_ids[i] != -1]
                local_centers = [np.array([shared_centers[2*i], shared_centers[2*i + 1]]) for i in range(len(local_ids))]
                local_corners_list = [np.array([shared_corners_list[8*i + j] for j in range(8)]).reshape(4, 2) for i in range(len(local_ids))]
                local_shared_velocity = np.array([shared_velocity[0], shared_velocity[1]])
                local_camera_read_time = last_camera_read_time.value
                #print(f"CAMERA read time: {local_camera_read_time:.3f} seconds")
                #print(f"shared_ids: {shared_ids[:10]}")  # Debugging output

            # Process detected markers (targets are all IDs != track_vel_id)
            for i, marker_id in enumerate(local_ids):
                if marker_id != track_vel_id:
                    center = local_centers[i]
                    corners = local_corners_list[i]
                    # Update or add to active_targets if not shot
                    if marker_id not in aruco_shooted:

                        if marker_id in active_targets:
                            active_targets[marker_id]['center'] = center
                            active_targets[marker_id]['corners'] = corners
                            active_targets[marker_id]['time'] = local_camera_read_time #current_time
                            #print(f"Updated active target ID {marker_id} at ({active_targets[marker_id]['center'][0]:.2f}, {active_targets[marker_id]['center'][1]:.2f})")
                        else:
                            active_targets[marker_id] = {
                                'center': center,
                                'corners': corners,
                                'time': local_camera_read_time,
                                'start_time': None  # Start time set when tracking begins
                            }
                            #print(f"Added new target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")
                    else:
                        # Update shot target position
                        aruco_shooted[marker_id]['center'] = center
                        aruco_shooted[marker_id]['corners'] = corners
                        aruco_shooted[marker_id]['time'] = local_camera_read_time #current_time
                        #print(f"Updated shot target ID {marker_id} at ({center[0]:.2f}, {center[1]:.2f})")

            #print(f"Difference of time : {time.perf_counter() - current_time}") # Result is ~1e-4 s
            # Extrapolate positions for active targets not detected
            detected_ids = set(local_ids)
            for marker_id in list(active_targets):
                if marker_id not in detected_ids:
                    elapsed = time.perf_counter() - active_targets[marker_id]['time'] #local_camera_read_time
                    active_targets[marker_id]['center'] += local_shared_velocity * elapsed
                    active_targets[marker_id]['corners'] += local_shared_velocity * elapsed
                    active_targets[marker_id]['time'] = time.perf_counter() #local_camera_read_time #current_time
            

            # Scanner update: track the prioritized target if any
            if active_targets:
                Start_counting_for_end = time.perf_counter()

                #if tracking_on:
                if tracking_on == True:
                    active_ids = list(active_targets.keys())
                    if local_shared_velocity[0] > pixel_threshold:
                        active_ids.sort(key=lambda mid: active_targets[mid]['center'][0], reverse=True)  # Right to left
                        #print("Prioritizing targets right-to-left (positive x-velocity).")
                    else:
                        active_ids.sort(key=lambda mid: active_targets[mid]['center'][0])  # Left to right
                        #print("Prioritizing targets left-to-right (negative or zero x-velocity).")
                    current_id = active_ids[0]
                    tracking_on = False
                current_target = active_targets[current_id]

                #print (f"Current target center : x: {current_target['center'][0]} , y: {current_target['center'][1]}")
                #print(f" ID {current_id} at x: (active_target : {active_targets[current_id]['center'][0]:.2f} | current target : {current_target['center'][0]:.2f}")

                # Set start_time when tracking begins
                if current_target['start_time'] is None:
                    current_target['start_time'] = current_time
                    print(f"Started tracking target ID {current_id} at ({current_target['center'][0]:.2f}, {current_target['center'][1]:.2f})")

                # Check if tracking duration exceeded
                if current_time - current_target['start_time'] > tracking_duration:
                    print(f"Finished tracking target ID {current_id} at ({current_target['center'][0]:.2f}, {current_target['center'][1]:.2f}). Moving to shot list.")
                    
                    aruco_shooted[current_id] = {
                        'center': current_target['center'],
                        'corners': current_target['corners'],
                        'time': local_camera_read_time,
                        'shot_time': current_time
                    }

                    del active_targets[current_id]
                    change_target = True
                    tracking_on = True 

                    """
                    # Sort active IDs by x-coordinate based on velocity direction
                    # changer la place du sort active ID (mettre dans la conditions de fin de tracking d'un aruco )
                    active_ids = list(active_targets.keys())
                    if local_shared_velocity[0] > pixel_threshold:
                        active_ids.sort(key=lambda mid: active_targets[mid]['center'][0], reverse=True)  # Right to left
                        #print("Prioritizing targets right-to-left (positive x-velocity).")
                    else:
                        active_ids.sort(key=lambda mid: active_targets[mid]['center'][0])  # Left to right
                        #print("Prioritizing targets left-to-right (negative or zero x-velocity).")
                    """
                    continue

                # Extrapolate current target's position
                elapsed = time.perf_counter() - current_target['time'] #local_camera_read_time current_time
                #print(f"ELAPSED TIME {elapsed:.6f} seconds.")
                extrapolated_center = current_target['center'] + local_shared_velocity * elapsed
                extrapolated_corners = current_target['corners'] + local_shared_velocity * elapsed  # Shape: (4, 2)


                #print(f"Current Target: {current_target['center']} | Displacement : {local_shared_velocity * elapsed}")
                extrapolated_x, extrapolated_y = extrapolated_center

                #print(f"current time: {current_time:.3f} seconds, active target time : {current_target['time']:.3f} seconds")
                #print(f"CAMERA read time: {local_camera_read_time:.3f} seconds")

                # Project to bits
                extrapolated_x_bit, extrapolated_y_bit = project_to_bits(
                    extrapolated_x, extrapolated_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale
                )

                if extrapolated_x_bit is None or extrapolated_y_bit is None:
                    print(f"Projection failed for extrapolated center ({extrapolated_x:.2f}, {extrapolated_y:.2f}). Skipping.")
                    continue
                
                #print(f"TIME SHOOT TARGET: {time.perf_counter():.3f} seconds")
                # Move scanner
                if change_target == True:
                    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
                    libe1701py.jump_abs(cardNum, extrapolated_x_bit, extrapolated_y_bit, 0)
                    libe1701py.execute(cardNum)
                    time.sleep(0.004) 
                    change_target = False
                    print(f"Test Jump to ID: {current_id}")
                #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                #libe1701py.execute(cardNum)
                draw_pattern_in_aruco(cardNum, "sandglass", extrapolated_corners, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
                
                libe1701py.execute(cardNum)
                lc.wait_marking(cardNum)
            else:
                ENDING_TIME = 3
                if time.perf_counter() - Start_counting_for_end > ENDING_TIME:
                    print(f"No active targets to track for {ENDING_TIME} seconds. END OF PROGRAM.")
                    break

                # No targets: Keep laser on but no move
                #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
                libe1701py.execute(cardNum)
                #print("No active targets to track.")
                
            # Sleep for ~0.0001s for fast scanner updates
            time.sleep(0.000001)
            
            process_time_main_loop = time.perf_counter()- current_time
            #print (f"process time MAIN LOOP = {process_time_main_loop} seconds")

    finally:
        libe1701py.close(cardNum)
        stop_event.set()
        cam_process.join()  # Ensure camera process is cleaned up

if __name__ == "__main__":

    unit = "pixels"  # or "mm"

    if unit == "mm":
        corr_file="corr_file_v4.bco"
    elif unit == "pixels":
        corr_file="corr_file_v5.bco"
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        exit(1)


    # Path to CSV file
    csv_path = "scanner_camera_map.csv" 


    

