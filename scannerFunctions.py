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


def close_all_devices(cardNum, picam2):

    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.close(cardNum)
    picam2.stop()
    cv2.destroyAllWindows()
    print("Laser off. Camera closed.")


def collect_pts_calib_scanner(unit="mm"):
    # ==== Parameters ====
    output_csv = "scanner_camera_map.csv"
    grid_size = 16  # nxn grid
    bit_range = 67108860//2 # full scanner field (±bit_rang)
    MAX_SPEED = 4294960000

    z = 0  # we assume 2D laser control 
    jump_speed = MAX_SPEED//10
    mark_speed = 50000
    laser_on_duration = 0.2  # seconds
    freq = 10000 #20000.0  # Hz, laser frequency

    # === ArUco marker config ===
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    reference_id = 23
    marker_length_mm = 24.0  # mm (2.4 cm)

    if unit == "mm":
        corr_file="corr_file_v4.bco"
    elif unit == "pixels":
        corr_file="corr_file_v5.bco"
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        exit(1)

    # ==== Initialize laser ====
    cardNum = lc.init_laser(port_name="/dev/ttyACM0", freq=freq, jump_speed=jump_speed, mark_speed=mark_speed,
                            corr_file=corr_file)

    print(" Laser initialized.")

    # ==== Initialize camera ====
    picam2, mtx, dist = pf.init_camera(gain = 1.0) #gain = 0.0 for default config
    print(" Camera initialized.")
    time.sleep(2)  # Allow camera to stabilize

    # ==== Define scan grid ====
    bit_positions = np.linspace(-bit_range, bit_range, grid_size, dtype=int)
    grid_points = [(x, y) for x in bit_positions for y in bit_positions]

    # ==== Output file ====
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        if unit == "mm":
            writer.writerow(["bit_x", "bit_y", "x_mm", "y_mm", "intensity_pixel"])

        elif unit == "pixels":
            #writer.writerow(["bit_x", "bit_y", "x_px", "y_px", "intensity_pixel"])
            writer.writerow(["bit_x", "bit_y", "x_px", "y_px", "spot_detected"])
        else:
            print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
            return

        print(" Starting calibration...")
        state_aruco_detected = False
        for idx, (bx, by) in enumerate(grid_points):
            print(f"[{idx+1}/{len(grid_points)}] Point: ({bx}, {by})")

            # Move laser to position
            libe1701py.jump_abs(cardNum, bx, by, z)
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)
            time.sleep(laser_on_duration)

            # Wait for marking to complete
            lc.wait_marking(cardNum)

            # Capture image & detect
            frame = picam2.capture_array()
            time.sleep(0.05)  
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if unit == "mm":
                if not state_aruco_detected:
                    print("   ➤ Detecting ArUco markers...")
                    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    if ids is not None and reference_id in ids:
                        state_aruco_detected = True
                        print("   ➤ ArUco marker detected.")
                    else:
                        print("   ➤ No ArUco marker detected. Retrying...")
                        continue

                #corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                x_mm, y_mm, px, py, intensity_pixel = pf.get_spot_coordinates(frame, corners, marker_length_mm, ids, reference_id)
                writer.writerow([bx, by, x_mm, y_mm, intensity_pixel]) 
                print(f"   ➤ Spot in world frame: ({x_mm}, {y_mm}) mm → Bit: ({px}, {py})")

            elif unit == "pixels":
                #px,py,intensity_pixel = pf.get_spot_coordinates_pixels(frame)
                #writer.writerow([bx, by, px, py, intensity_pixel]) 
                spot_pt,spot_detected = pf.detect_laser_spot2(frame)
                px = spot_pt [0]
                py = spot_pt [1]
                writer.writerow([bx, by, px, py, spot_detected])
                print(f"   ➤ Spot in pixel frame: ({px}, {py}) pixels → Bit: ({bx}, {by})")
            else:
                print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
                return

            cv2.circle(frame, (px, py),10, (0, 0, 255), 2)
            #aruco.drawDetectedMarkers(frame, corners, ids)
            small = cv2.resize(frame, (640, 480))  
            cv2.imshow("Preview Calibration", small)
            #cv2.imshow("Laser Spot Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Turn off laser
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
            time.sleep(0.1)

    # ==== Cleanup ====
    print("Calibration data saved to:", output_csv)
    close_all_devices(cardNum, picam2)


def load_data(unit = "mm",csv_path="scanner_camera_map.csv", SPOT_INTENSITY_TRESHOLD=220):
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


def plot_points(xy, bit_xy, unit="mm"):
    """Plot the (x_mm, y_mm) and (bit_x, bit_y) points."""
    plt.figure(figsize=(12, 6))
    
    # Plot (x_mm, y_mm)
    plt.subplot(1, 2, 1)
    plt.scatter(xy[:, 0], xy[:, 1], c='blue', label='xy')
    if unit == "mm":
        plt.xlabel('x_mm')
        plt.ylabel('y_mm')
    elif unit == "pixels":
        plt.xlabel('x_px')
        plt.ylabel('y_px')
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        return
    plt.title('Points (x_mm, y_mm)' if unit == "mm" else 'Points (x_px, y_px)')
    plt.legend()
    
    # Plot (bit_x, bit_y)
    plt.subplot(1, 2, 2)
    plt.scatter(bit_xy[:, 0], bit_xy[:, 1], c='red', label='bit_xy')
    plt.xlabel('bit_x')
    plt.ylabel('bit_y')
    plt.title('Points (bit_x, bit_y)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_points_mm_bits(xy,bit_xy, unit="mm"):
    """Plot the (x_mm, y_mm) and (bit_x, bit_y) points."""
    plt.figure(figsize=(12, 6))
    
    # Plot (x, bit_x)
    plt.subplot(1, 2, 1)
    plt.scatter(xy[:, 0], bit_xy[:, 0], c='blue', label='x-axis')
    if unit == "mm":
        plt.xlabel('x_mm')
        plt.ylabel('x_bits')
    elif unit == "pixels":
        plt.xlabel('x_px')
        plt.ylabel('x_bits')    
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        return
    plt.title('Points (x_mm, x_bits)' if unit == "mm" else 'Points (x_px, x_bits)')
    plt.legend()

    # Plot (y, bit_y)
    plt.subplot(1, 2, 2)
    plt.scatter(xy[:, 1], bit_xy[:, 1], c='red', label='y-axis')
    if unit == "mm":
        plt.xlabel('y_mm')
        plt.ylabel('y_bits')
    elif unit == "pixels":
        plt.xlabel('y_px')
        plt.ylabel('y_bits')
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        return 
    plt.title('Points (y_mm, y_bits)' if unit == "mm" else 'Points (y_px, y_bits)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



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
        [0, 0],  # Bottom-left
        [0, 1],  # Top-left
        [1, 1],  # Top-right
        [1, 0]   # Bottom-right
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
    else:
        print(f"Unknown pattern: {pattern_name}. Supported patterns: square, spiral, sandglass, zigzag.")
        return

    # Convert center to bits
    x_center, y_center = project_to_bits(center_x, center_y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
    if x_center is None or y_center is None:
        print(f"Projection failed for the center ({center_x}, {center_y}).")
        return

    # If aruco has a constant speed, then modify goal_pts accordingly
    if (marking_speed != 0) and (aruco_speed != 0):
        goal_pts = create_moving_goal_pts(goal_pts, aruco_speed, marking_speed)
    
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
    for x_bit, y_bit in goal_pts_bits:
        libe1701py.mark_abs(cardNum, x_bit, y_bit, 0)

    #libe1701py.execute(cardNum)
    #lc.wait_marking(cardNum)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")

def draw_in_aruco_during_time(cardNum, center_x,center_y,pattern_name, corners,shoot_time = 0.0025,  xy_norm=None, bit_xy_norm=None, min_coord=None, max_coord=None, bit_scale=None, aruco_speed=0, marking_speed = 0):
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


def shoot_target_by_priority_px(cardNum, picam2, csv_path="scanner_camera_map.csv"):
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
    MID_THRESHOLD = 60 #120
    coeff_time = 0.01 # 0.01 for ms units 0.3

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

        #go_to_several_points_without_cam(cardNum, points,shooting_time = 0.005, unit = "pixels", csv_path="scanner_camera_map.csv")
        
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


def live_tracking_px_predict(cardNum, picam2, track_id, csv_path="scanner_camera_map.csv"):
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



def live_tracking_px_v2(cardNum, picam2, track_id, csv_path="scanner_camera_map.csv"):
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = load_data(unit, csv_path)
    pixel_threshold = 2  # Threshold to detect movement
    camera_update_interval = 0.1  # Update camera every 0.5 seconds

    # Initialize tracking variables
    last_center = None
    last_velocity = np.array([0.0, 0.0])  # Initial velocity
    last_update_time = None
    last_camera_read_time = time.perf_counter() - camera_update_interval  # Force initial read

    # Turn on the laser at initial position
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)

    while True:
        current_time = time.perf_counter()

        # Check if it's time to update from camera (every 0.5s)
        if current_time - last_camera_read_time >= camera_update_interval:
            last_camera_read_time = current_time

            frame = picam2.capture_array()
            centers, ids, corners_list, frame = pf.detect_aruco_markers(frame, unit)

            small = cv2.resize(frame, (640, 480))
            cv2.imshow("Live Camera Feed", small)

            if track_id in ids:
                index = np.where(ids == track_id)[0][0]
                corners = corners_list[index].reshape(4, 2)
                current_center = np.mean(corners, axis=0)  # [x, y]

                # If first detection, initialize
                if last_center is None:
                    last_center = current_center
                    last_update_time = current_time
                    print(f"Initial detection of ID {track_id} at ({current_center[0]:.2f}, {current_center[1]:.2f})")
                    continue

                # Calculate dt and velocity
                dt = current_time - last_update_time
                if dt > 0:
                    last_velocity = (current_center - last_center) / dt
                else:
                    last_velocity = np.array([0.0, 0.0])

                # Update last known position and time
                last_center = current_center
                last_update_time = current_time

                print(f"Camera update: ID {track_id} at ({current_center[0]:.2f}, {current_center[1]:.2f}), velocity ({last_velocity[0]:.2f}, {last_velocity[1]:.2f})")

                # Check if stationary
                if abs(current_center[0] - last_center[0]) < pixel_threshold and abs(current_center[1] - last_center[1]) < pixel_threshold:
                    # Stationary: No need to move, but keep laser on
                    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
                    libe1701py.execute(cardNum)
                    continue
            else:
                # Target lost: Keep using last velocity/position for extrapolation, or handle as needed
                print(f"ID {track_id} not detected during camera update. Continuing with last known velocity.")
                # Optionally reset velocity to 0 if lost for too long, but for now continue

        # Extrapolate position for scanner update (every loop ~0.0001s, but in practice faster)
        if last_center is not None and last_update_time is not None:
            elapsed = current_time - last_update_time
            extrapolated_center = last_center + last_velocity * elapsed
            extrapolated_x, extrapolated_y = extrapolated_center

            # Project extrapolated position to bits
            extrapolated_x_bit, extrapolated_y_bit = project_to_bits(
                extrapolated_x, extrapolated_y, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale
            )

            if extrapolated_x_bit is None or extrapolated_y_bit is None:
                print(f"Projection failed for extrapolated center ({extrapolated_x:.2f}, {extrapolated_y:.2f}). Skipping.")
                continue

            # Move scanner to extrapolated position
            libe1701py.mark_abs(cardNum, extrapolated_x_bit, extrapolated_y_bit, 0)
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

if __name__ == "__main__":

    unit = "pixels"  # or "mm"

    if unit == "mm":
        corr_file="corr_file_v4.bco"
    elif unit == "pixels":
        corr_file="corr_file_v5.bco"
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        exit(1)

    #Collect calibration points (if needed)
    collect_pts_calib_scanner(unit)

    # Path to CSV file
    csv_path = "scanner_camera_map.csv" 
    #disc_first_n = 0

    xy_coord, bit_xy, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale = load_data(unit, csv_path)
    
    # Plot points for visualization
    plot_points(xy_coord, bit_xy, unit)
    plot_points_mm_bits(xy_coord, bit_xy, unit)

    cardNum=lc.init_laser(port_name="/dev/ttyACM0", freq=10000, jump_speed=4294960000//10, mark_speed=5000,
                        corr_file=corr_file)

    #Square drawing test
    px_test = 1000
    py_test = 800
    draw_square_world_frame(cardNum, 300,px_test,py_test, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale)

    # Go to several arbitrary points test
    points = [
        (30, 0),
        (30, 30),
        (0, 30),
        (0, 0)
    ]
   # go_to_several_points_without_cam(cardNum, points,unit)

