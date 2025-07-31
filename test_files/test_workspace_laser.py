import libe1701py
import time

# Define the full field range (limited by the protocol XY2-100)
max_coord = 131000
min_coord = -131000
jump_speed = 2000. #Max : 262140
mark_speed = 500. #Max : 262140

z = 0

# Connect to the card
cardNum = libe1701py.set_connection("/dev/ttyACM0")
if cardNum > 0:
    ret = libe1701py.load_correction(cardNum, "", 0)
    if ret == libe1701py.E170X_OK:
        libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)
        libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2
        )
        libe1701py.set_laser_timing(cardNum, 2*20000.0, 50.0) #25.0 for 50% duty cycle

        libe1701py.set_speeds(cardNum, jump_speed , mark_speed ) # 67108864.0 , 67108.864 jump_speed, mark_speed

        libe1701py.jump_abs(cardNum, min_coord, min_coord, z)
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")

        libe1701py.mark_abs(cardNum, min_coord, min_coord, z)
        libe1701py.mark_abs(cardNum, min_coord, max_coord, z)
        libe1701py.mark_abs(cardNum, max_coord, max_coord, z)
        libe1701py.mark_abs(cardNum, max_coord, min_coord, z)
        libe1701py.mark_abs(cardNum, min_coord, min_coord, z)
        libe1701py.mark_abs(cardNum, 0, 0,z)

        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
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


        libe1701py.close(cardNum)
    else:
        print("ERROR: Correction file load failed.")
else:
    print("ERROR: Could not initialize connection.")
