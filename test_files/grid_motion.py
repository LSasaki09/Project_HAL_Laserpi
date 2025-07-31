import cv2
import numpy as np
import time
import libe1701py


grid_size = 6  # 5x5 grid
bit_range = 67108860 //2 #1000000  # full scanner field (±bit_rang)
MAX_SPEED = 4294960000

z = 0  # we assume 2D laser control
jump_speed = MAX_SPEED//10
mark_speed = 50000
laser_on_duration = 0.3  # seconds
freq = 10000 #20000.0  # Hz, laser frequency


# ==== Define scan grid ====
bit_positions = np.linspace(-bit_range, bit_range, grid_size, dtype=int)
grid_points = [(x, y) for x in bit_positions for y in bit_positions]

# ==== Setup scanner ====
cardNum = libe1701py.set_connection("/dev/ttyACM0")
libe1701py.load_correction(cardNum, "corr_file_v3.bco", 0)
libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)
#libe1701py.set_scanner_delays2(cardNum, 0, 100.0, 100.0, 10.0)
libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)
libe1701py.set_laser_timing(cardNum, freq, 50.0)
libe1701py.set_speeds(cardNum, jump_speed, mark_speed)


for idx, (bx, by) in enumerate(grid_points):
        print(f"[{idx+1}/{len(grid_points)}] Point: ({bx}, {by})")

        # Move laser to position
        libe1701py.jump_abs(cardNum, bx, by, z)
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
        libe1701py.execute(cardNum)
        time.sleep(laser_on_duration)

        # Turn off laser
        #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
        #libe1701py.execute(cardNum)
        #time.sleep(laser_on_duration)

        state = libe1701py.get_card_state(cardNum)
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) == 0:
            time.sleep(0.01)
            state = libe1701py.get_card_state(cardNum)

        # Wait for marking to complete
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) != 0:
            time.sleep(0.05)
            state = libe1701py.get_card_state(cardNum)

        

        print(f"Point ({bx}, {by}) processed.")
        time.sleep(0.1)


libe1701py.close(cardNum)