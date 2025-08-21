import csv
import pandas as pd
import laserControl as lc
import picamFunctions as pf
import shootingFunctions as sf
import cv2
from cv2 import aruco
from picamera2 import Picamera2
import numpy as np
import time
import timeit
import libe1701py
import matplotlib.pyplot as plt 

def analyze_delays(delays, label):
    """Calculate and print statistical metrics for delay measurements."""
    delays = np.array(delays)
    mean_delay = np.mean(delays)
    std_delay = np.std(delays)
    min_delay = np.min(delays)
    max_delay = np.max(delays)
    median_delay = np.median(delays)
    print(f"\n{label} Delay Statistics:")
    print(f"  Mean: {mean_delay:.6f} seconds")
    print(f"  Standard Deviation: {std_delay:.6f} seconds")
    print(f"  Min: {min_delay:.6f} seconds")
    print(f"  Max: {max_delay:.6f} seconds")
    print(f"  Median: {median_delay:.6f} seconds")

def measure_communication_delay(cardNum, picam2,nb_data_points=100):
    time_stamps = np.zeros(nb_data_points)
    #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
    i = 1
    start = time.perf_counter()
    prev_time = start

    libe1701py.jump_abs(cardNum, 0, 0, 0)

    spot_intensity = 0 

    while i <= nb_data_points-1:
        #frame = picam2.capture_array()
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #_, spot_intensity = pf.detect_laser_spot(frame)
        #print(f"Spot intensity: {spot_intensity}")

        if spot_intensity < 230: #means laser turned off

            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, '1') #turn on
            #libe1701py.execute(cardNum)

            while spot_intensity < 240:
                #print("Waiting for laser to turn on...")
                
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, spot_intensity = pf.detect_laser_spot(frame)

            #time_stamps[i] = time.perf_counter() - time_stamps[i-1] - start
            current_time = time.perf_counter()
            time_stamps[i] = current_time - prev_time
            prev_time = current_time

        else: #means laser turned on
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, '0') #turn off
            #libe1701py.jump_abs(cardNum, 0, 0, 0) #jump to 0,0,0
            libe1701py.execute(cardNum)
            #print("Laser turned off")

            while spot_intensity > 240:
                
                #print("Waiting for laser to turn off...")
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, spot_intensity = pf.detect_laser_spot(frame)

           # time_stamps[i] = time.perf_counter() - time_stamps[i-1] - start 
            current_time = time.perf_counter()
            time_stamps[i] = current_time - prev_time
            prev_time = current_time
        i += 1

    analyze_delays(time_stamps, "Communication")
    
    return time_stamps

def measure_camera_delay(cardNum, picam2, mtx_0, dist_0, num_measurements=100):
    """Measure the delay between consecutive camera frame captures."""
    delays = []
    """
    # Turn on laser and stabilize
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, '1')
    libe1701py.execute(cardNum)
    time.sleep(1)  # Wait for laser to stabilize
    """

    frame = picam2.capture_array()
    start = time.perf_counter()
    prev_time = start
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i in range(num_measurements):
        _ = pf.detect_aruco_markers(frame, mtx=mtx_0, dist=dist_0, marker_length=24.0)
        #print(f"Iteration {i+1}: Spot location = {loc}, Spot intensity = {spot_intensity}")

        current_time = time.perf_counter()
        delay = current_time - prev_time
        delays.append(delay)
        prev_time = current_time

    analyze_delays(delays, "Camera")
    return delays

def measure_laser_delay(cardNum,num_measurements=100):
    """Measure the delay between laser commands."""
    delays_on = []
    delays_off = []

    libe1701py.jump_abs(cardNum, 0, 0, 0)
    time.sleep(0.1) 

    state_on = False

    start = time.perf_counter()
    current_time = start

    for i in range(num_measurements):
        if i % 2 == 0:
            state_on = False
        else:
            state_on = True

        if state_on==False:
            prev_time = time.perf_counter()
            ret = libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, '1')
            while ret != 0:
                time.sleep(0.00000001)
            current_time = time.perf_counter()
            delay = current_time - prev_time
            delays_on.append(delay)

        else:
            prev_time = time.perf_counter()
            ret = libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, '0')
            libe1701py.execute(cardNum)
            while ret != 0:
                time.sleep(0.0000001)
            current_time = time.perf_counter()
            delay = current_time - prev_time
            delays_off.append(delay)
        
        time.sleep(0.05)  # Wait before next command

    print("Laser On Delays: ", delays_on)
    print("Laser Off Delays: ", delays_off)

    # Analyze delays
    print("\nAnalyzing delays for laser on and off commands:")

    analyze_delays(delays_on, "Laser On")
    analyze_delays(delays_off, "Laser Off")

    return delays_on, delays_off

def plot_speed_relationship(speeds_bits, speeds_px, slope, intercept):
    """
    Trace la relation entre vitesse_bits et vitesse_px avec la droite d'interpolation.
    
    Paramètres :
    - speeds_bits : Liste des vitesses en bits/s.
    - speeds_px : Liste des vitesses en pixels/s.
    - slope : Pente de la relation linéaire.
    - intercept : Ordonnée à l'origine de la relation linéaire.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(speeds_bits, speeds_px, c='blue', label='Données mesurées')
    
    # Tracer la droite d'interpolation
    x_fit = np.array(speeds_bits)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit linéaire: y = {slope:.6f}x + {intercept:.6f}')
    
    plt.xlabel('Vitesse (bits/s)')
    plt.ylabel('Vitesse (pixels/s)')
    plt.title('Relation entre vitesse en bits/s et vitesse en pixels/s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calibrate_laser_speed(cardNum, picam2, start_px=(600, 100), end_px=(600, 1200), speeds_bits=None, num_samples=100, csv_path="scanner_camera_map.csv", output_csv="speed_calibration.csv"):
    """
    Calibre la vitesse du laser en établissant une relation entre vitesse en bits/s et vitesse en pixels/s.
    
    """
    MAX_SPEED_BITS = 4294960000 
    MAX_SPEED_BITS_MARK = 4294960000 // 100000  
    INTENSITY_THRESHOLD = 230  # Seuil d'intensité pour filtrer les points

    if speeds_bits is None:
        #speeds_bits = np.logspace(3, np.log10(MAX_SPEED_BITS_MARK), num=8, dtype=int)  # Plage logarithmique de vitesses
        speeds_bits = np.linspace(1000, MAX_SPEED_BITS_MARK, num=15, dtype=int)  # Plage linéaire de vitesses

    # Charger les données de calibration de position (en pixels)
    unit = "pixels"
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = sf.load_data(unit, csv_path)

    # Projeter les points de départ et d'arrivée en bits
    start_bit_x, start_bit_y = sf.project_to_bits(start_px[0], start_px[1], coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
    end_bit_x, end_bit_y = sf.project_to_bits(end_px[0], end_px[1], coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)

    if start_bit_x is None or start_bit_y is None or end_bit_x is None or end_bit_y is None:
        print("Erreur : Impossible de projeter les points de départ ou d'arrivée en bits.")
        return None, None

    # Liste pour stocker les résultats (vitesse_bits, vitesse_pixels)
    results = []
    i = 1
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speed_bits", "speed_px"])

        for speed_bits in speeds_bits:
            print(f"Test de la vitesse : {speed_bits} bits/s")

            # Positionner le laser au point de départ
            libe1701py.jump_abs(cardNum, start_bit_x, start_bit_y, 0)
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
            libe1701py.execute(cardNum)

            # Déplacer le laser vers le point d'arrivée à la vitesse donnée
            libe1701py.set_speeds(cardNum, MAX_SPEED_BITS // 10, speed_bits)
            libe1701py.mark_abs(cardNum, end_bit_x, end_bit_y, 0)
            libe1701py.execute(cardNum)

            # Capturer les positions, timestamps et intensités pendant le mouvement
            positions = []
            timestamps = []
            intensities = []
            start_time = time.perf_counter()
            n = 0
            while True:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                px, py, intensity = pf.get_spot_coordinates_pixels(frame)

                if px is not None and py is not None and intensity >= INTENSITY_THRESHOLD:
                    positions.append((px, py))
                    timestamps.append(time.perf_counter() - start_time)
                    intensities.append(intensity)

                n += 1

                if n >= num_samples:
                    lc.wait_marking(cardNum)  # Attendre la fin du marquage
                    break
                #time.sleep(1 / (np.power(10,i)))  # Ajuster selon la fréquence de capture souhaitée
                #time.sleep(1/(10*i))  # Ajuster la fréquence de capture
                time.sleep(0.01)  # Ajuster la fréquence de capture

            i += 1
            print(f"Captured {len(positions)} positions at speed {speed_bits} bits/s")
            print(f"Positions : {positions}")
            

            
            # Calculer la vitesse moyenne en pixels/s avec les données filtrées
            if len(positions) >= 2:
                distances = np.sqrt(np.diff([p[0] for p in positions])**2 + np.diff([p[1] for p in positions])**2)
                times = np.diff(timestamps)
                speeds_px = distances / times
                avg_speed_px = np.mean(speeds_px)
                results.append((speed_bits, avg_speed_px))
                writer.writerow([speed_bits, avg_speed_px])
                print(f"Elapsed time : {times}")
                print(f"Vitesse moyenne : {avg_speed_px:.2f} pixels/s (points valides : {len(positions)})")
            else:
                print("Pas assez de données valides pour calculer la vitesse après filtrage.")

            # Éteindre le laser
            libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
            libe1701py.execute(cardNum)
            time.sleep(0.1)
 
def load_speeds_calib(csv_path="speed_calibration.csv"):
    """
    Charge les données de calibration de vitesse et calcule la relation linéaire.
    
    """
    df = pd.read_csv(csv_path)
    speeds_bits = df['speed_bits'].values
    speeds_px = df['speed_px'].values
    coefficients = np.polyfit(speeds_bits, speeds_px, 1)
    slope, intercept = coefficients
    print(f"Relation linéaire chargée : vitesse_px = {slope:.6f} * vitesse_bits + {intercept:.6f}")

    plot_speed_relationship(speeds_bits, speeds_px, slope, intercept)

    return slope, intercept

def convert_px_to_bits_speed(speed_px, slope, intercept):
    """
    Convertit une vitesse en pixels/s en vitesse en bits/s.
    
    Paramètres :
    - speed_px : Vitesse en pixels/s.
    - slope : Pente de la relation linéaire.
    - intercept : Ordonnée à l'origine de la relation linéaire.
    
    Retourne :
    - speed_bits : Vitesse en bits/s.
    """
    if slope == 0:
        raise ValueError("La pente ne peut pas être zéro pour la conversion.")
    speed_bits = (speed_px - intercept) / slope
    return speed_bits

def laser_accuracy(cardNum, picam2, m, d):
    """
    Determines Laser accuracy by comparing aruco center to laser spot position
    """
    coord_xy, bit_xy, coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale = sf.load_data("pixels")

    libe1701py.jump_abs(cardNum, 0, 0, 0)
    old_pos = [0, 0]
    current_pos = [0, 0]
    frame = picam2.capture_array()
    centers, _, _, _ = pf.detect_aruco_markers(frame, mtx = m, dist = d)
    print(centers)
    
    pf.modify_exposure_and_gain(picam2, exposure = 30000, gain = 1.0)

    errors = []
    for center in centers:
        x_bit, y_bit = sf.project_to_bits(center[0], center[1], coord_xy_norm, bit_xy_norm, coord_min, coord_max, bit_scale)
        libe1701py.jump_abs(cardNum, x_bit, y_bit, 0)
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
        libe1701py.execute(cardNum)
        time.sleep(0.5)
        frame = picam2.capture_array()
        #laser_pos, _ = pf.detect_laser_spot(frame)
        laser_pos, spot_detected = pf.detect_laser_spot2(frame, test = False)
        time.sleep(0.1)
        diff = np.asarray(center) - np.asarray(laser_pos)
        error = np.linalg.norm(diff)
        errors = np.append(errors, error)
        print(f"Error = {error} pixels | Laser spot = {laser_pos}")
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"Mean Error = {mean_error} pixels | Standard deviation = {std_error}")
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.execute(cardNum)
    """
    small = cv2.resize(frame, (640, 480))
    cv2.imshow("Screen", small)

    # Wait until spacebar (ASCII 32) is pressed
    
    while True:
        if cv2.waitKey(1) & 0xFF == 32:  # spacebar
                libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
                libe1701py.execute(cardNum)
                break
    
    cv2.destroyAllWindows()
    """

def Study_beam_profile(picam2, cardNum, exp = 30000, g = 1.0, n = 300):
    pf.modify_exposure_and_gain(picam2, exposure = exp, gain = g)
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)

    frame = picam2.capture_array()
    laser_spot, detect = pf.detect_laser_spot2(frame)
    if detect == True:
        cropped_frame = frame[laser_spot[1]-n:laser_spot[1]+n, laser_spot[0]-n:laser_spot[0]+n]
        cv2.imshow("Focuse on beam spot", cropped_frame)
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  # spacebar
                break
        cv2.destroyAllWindows()
    else :
        print("No spot detected")
    
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.execute(cardNum)

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
    
    elif pattern_name == "simple_zigzag":
        x, y = corners[1].tolist()[0], corners[1].tolist()[1]
        x1, y1 = sf.project_to_bits(x, y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        x, y = corners[2].tolist()[0], corners[2].tolist()[1]
        x2, y2 = sf.project_to_bits(x, y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
        if y1 is None or y2 is None:
            Max_wobble_amplitude = 30000000
        else:
            Max_wobble_amplitude = np.absolute(y1 - y2)
        libe1701py.set_wobble(cardNum, Max_wobble_amplitude, Max_wobble_amplitude, 250)
        start_point, end_point = (unit_square[0] + unit_square[1])/2, (unit_square[2] + unit_square[3])/2
        goal_pts_unit = [start_point, end_point]
        goal_pts_unit = np.array(goal_pts_unit, dtype=np.float32).reshape(-1, 1, 2)
        goal_pts = cv2.perspectiveTransform(goal_pts_unit, h_matrix).reshape(-1, 2).tolist()

    else:
        print(f"Unknown pattern: {pattern_name}. Supported patterns: square, spiral, sandglass, zigzag, simple_zigzag.")
        return

    # Convert center to bits
    x_center, y_center = sf.project_to_bits(center_x, center_y, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
    if x_center is None or y_center is None:
        print(f"Projection failed for the center ({center_x}, {center_y}).")
        return

    # If aruco has a constant speed, then modify goal_pts accordingly
    if (marking_speed != 0) and (aruco_speed != 0):
        goal_pts = create_moving_goal_pts(goal_pts, aruco_speed, marking_speed)
    
    # Convert pattern points to laser coordinates (bits)
    goal_pts_bits = []
    for x_g, y_g in goal_pts:
        x_bit, y_bit = sf.project_to_bits(x_g, y_g, xy_norm, bit_xy_norm, min_coord, max_coord, bit_scale)
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


if __name__ == "__main__":
    picam2, m, d = pf.init_camera()
    time.sleep(0.5)
    cardNum = lc.init_laser()


    """
    - All these are different codes to estimate sources of delays and lag

    #nb_data_points = 100 # Number of data points for delay measurement
    #frame = picam2.capture_array()
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #execution_time = timeit.timeit("picam2.capture_array()", number=nb_data_points, globals=globals())
    #execution_time = timeit.timeit("pf.detect_aruco_markers(frame, mtx=mtx_0, dist=dist_0, marker_length=24.0)", number=nb_data_points, globals=globals())
    #execution_time = timeit.timeit("pf.detect_laser_spot(frame)", number=nb_data_points, globals=globals())
    #print(f"Average execution time of : {execution_time / nb_data_points:.6f} s")
    #comm_delays = measure_communication_delay(cardNum, picam2,nb_data_points)
    #cam_delays = measure_camera_delay(cardNum, picam2, mtx, dist, nb_data_points)

    #measure_laser_delay (cardNum, nb_data_points)
    #print(f"Communication delays: {comm_delays}")
    #print(f"Camera delays: {cam_delays}")
    """

    #calibrate_laser_speed(cardNum, picam2)
    #slope,intercept = load_speeds_calib("speed_calibration.csv")

    """
    #It's Wobbling time !!
    Marking_coord = [67108860//5, 0]
    Mark_speed = 10000
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.execute(cardNum)
    lc.Woobling_time(cardNum, Marking_coord, marking_speed = Mark_speed)
    """
    """
    # Test for detect_laser_spot2()
    libe1701py.jump_abs(cardNum, 0, 0, 0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    libe1701py.execute(cardNum)
    for i in range(10):
        frame = picam2.capture_array()
        spot = pf.detect_laser_spot2(frame)
        print(spot)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.execute(cardNum)
    """

    # Test to estimate the laser's pointpoint accuracy
    """_
    laser_accuracy(cardNum, picam2, m = mtx_0, d = dist_0)
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")
    libe1701py.execute(cardNum)
    _"""

    # Study beam profile
    """
    Study_beam_profile(picam2, cardNum, exp = 8000, g = 1.0, n = 200)
    """

    # Test to understand what happens to detect_laser_spot2 if there are no laser
    """
    for i in range(10):
        frame = picam2.capture_array()
        time.sleep(0.5)
        loc, spot_detected = pf.detect_laser_spot2(frame)
        print(loc), print(spot_detected)
    """

    # Test for Draw pattern
    MAX_SPEED_BITS_MARK = 4294960000 // 200000
    libe1701py.set_speeds(cardNum, MAX_SPEED_BITS_MARK, MAX_SPEED_BITS_MARK)
    frame = picam2.capture_array()
    pts_xy, bit_xy, xy_n, bit_xy_n, pt_min, pt_max, bit_s = sf.load_data()
    centers, ids, corners, frame = pf.detect_aruco_markers(frame, mtx = m, dist = d)
    time.sleep(1)

    track_id = 0
    if track_id in ids:
        index_track_id = np.where(ids == track_id)[0][0]
        corners_track_id = corners[index_track_id].reshape(4, 2)
        print(f"corners track id 0: {corners_track_id}")
        draw_pattern_in_aruco(cardNum, "simple_zigzag", corners_track_id, xy_norm=xy_n, bit_xy_norm=bit_xy_n, min_coord=pt_min, max_coord=pt_max, bit_scale=bit_s)
        libe1701py.execute(cardNum)
        lc.wait_marking(cardNum)

    sf.close_all_devices(cardNum, picam2)