import libe1701py
import time
import numpy as np

# Central point and square parameters
x0, y0, z0 = 0, 0, 0  # Central point 

square_size = 67108860 #262140 #2328550 Define the full field range (limited by the protocol XY2E-100)
jump_speed = 20000. #Max : 262140
mark_speed = 50000. #Max : 262140


# Establish connection
cardNum = libe1701py.set_connection("/dev/ttyACM0")
if cardNum > 0:
    # Load a correction file (empty for no correction)
    ret = libe1701py.load_correction(cardNum, "corr_file_v4.bco", 0)
    print(f"Correction file load result: {ret}")
    if ret == libe1701py.E170X_OK:
        # Configure the scan head (XY3-100 mode, 20-bit precision)
        # libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY3_20BIT)
        libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)  # for XY2E-100 protocol (18-bit precision)
        
        libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)
        libe1701py.set_laser_timing(cardNum, 20000.0, 50.0)

        libe1701py.set_speeds(
            cardNum,
            jump_speed,
            mark_speed
        )  # Max speed for jump and mark

        # Move the laser to the central point (laser off)
        libe1701py.jump_abs(cardNum, x0, y0, z0)

        # Turn on the laser
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")

        # Draw a square around the central point
        half_size = square_size //2
        libe1701py.mark_abs(cardNum, x0 - half_size, y0 - half_size, z0)  # Lower left corner
        libe1701py.mark_abs(cardNum, x0 - half_size, y0 + half_size, z0)  # Upper left corner
        libe1701py.mark_abs(cardNum, x0 + half_size, y0 + half_size, z0)  # Upper right corner
        libe1701py.mark_abs(cardNum, x0 + half_size, y0 - half_size, z0)  # Lower right corner
        libe1701py.mark_abs(cardNum, x0 - half_size, y0 - half_size, z0)  # Back to the starting corner
        libe1701py.mark_abs(cardNum, 0, 0,z0)

        
        time.sleep(0.2)
        #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
        
        # Start executing the commands
        libe1701py.execute(cardNum)

        # Wait for marking to start
        state = libe1701py.get_card_state(cardNum)
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) == 0:
            time.sleep(0.01)
            state = libe1701py.get_card_state(cardNum)

        # Wait for marking to complete
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) != 0:
            time.sleep(0.1)
            state = libe1701py.get_card_state(cardNum)

        # Retrieve the last position
        time.sleep(0.5)
        ret, x, y, z = libe1701py.get_pos(cardNum)
        if ret == libe1701py.E170X_OK:
            print(f"Last position: x={x}, y={y}, z={z}")
        else:
            print(f"ERROR: get_pos failed with error {ret}")

        # Close the connection
        time.sleep(0.1)
        libe1701py.close(cardNum)
    else:
        print(f"ERROR: opening correction file failed with error {ret}")
else:
    print("ERROR: Could not initialize connection!")
