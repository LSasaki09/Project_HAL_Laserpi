import cv2
import numpy as np
import yaml
from picamera2 import Picamera2
import libe1701py
import time
from cv2 import aruco
import picamFunctions as pf


# === Load camera calibration ===
with open('calibration.yaml') as f:
    calib = yaml.load(f, Loader=yaml.FullLoader)
mtx = np.array(calib['camera_matrix'])
dist = np.array(calib['dist_coeff'])

# === ArUco marker config ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
reference_id = 23
marker_length_mm = 24.0  # mm (2.4 cm)

# === Camera init ===
picam2 = Picamera2()
config = picam2.create_still_configuration(
        main={"size": ( 2028,1520)},  # 120 fps, 1332x990 resolution / 2028×1080 50 fps, 2028×1520 40 fps
        buffer_count=1
    )
picam2.configure(config)

print(picam2.sensor_modes)

picam2.start()
time.sleep(1)

# === Connect to scanner and enable laser ===
cardNum = libe1701py.set_connection("/dev/ttyACM0")
if cardNum <= 0:
    print(" Could not connect to scanner.")
    exit(1)

libe1701py.load_correction(cardNum, "corr_file_v4.bco", 0)
libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)
libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)
libe1701py.set_laser_timing(cardNum, 10000.0, 50.0)
libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
libe1701py.execute(cardNum)

print("🟢 Laser is ON.")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(corners) > 0:
           X_mm, Y_mm, px, py,_ = pf.get_spot_coordinates (frame, corners, marker_length_mm, ids, reference_id)
           cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
        else:
            print("No markers detected.")

        aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Laser Spot Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
    libe1701py.close(cardNum)
    picam2.stop()
    cv2.destroyAllWindows()
    print("Laser off. Camera closed.")

