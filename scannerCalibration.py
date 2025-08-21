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
import shootingFunctions as sf



def collect_pts_calib_scanner(unit="pixels"):
    # ==== Parameters ====
    output_csv = "scanner_camera_map.csv"
    grid_size = 17  # nxn grid
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
    sf.close_all_devices(cardNum, picam2)

def plot_points(xy, bit_xy, unit="pixels"):
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

def plot_points_mm_bits(xy,bit_xy, unit="pixels"):
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

if __name__ == "__main__":
    unit = "pixels"
    collect_pts_calib_scanner(unit)
    # Path to CSV file
    csv_path = "scanner_camera_map.csv" 
    #disc_first_n = 0
    xy_coord, bit_xy, xy_norm, bit_xy_norm, pt_min, pt_max, bit_scale = sf.load_data(unit, csv_path)
    plot_points(xy_coord, bit_xy, unit)
    plot_points_mm_bits(xy_coord, bit_xy, unit)


    #cardNum=lc.init_laser(port_name="/dev/ttyACM0", freq=10000, jump_speed=4294960000//10, mark_speed=50000,
       #                 corr_file=corr_file)