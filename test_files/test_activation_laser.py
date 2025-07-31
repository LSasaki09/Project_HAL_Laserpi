import libe1701py
import time

coord = -131000 //100
jump_speed = 2000. #Max : 262140
mark_speed = 500. #Max : 262140

cardNum = libe1701py.set_connection("/dev/ttyACM0")

if cardNum > 0:
    libe1701py.load_correction(cardNum, "", 0)
    libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)


    libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)
    libe1701py.set_laser_timing(cardNum, 20000.0, 25.0)
    libe1701py.set_speeds(cardNum, jump_speed , mark_speed ) # 67108864.0 , 67108.864 jump_speed, mark_speed

    #libe1701py.jump_abs(cardNum, coord,-coord , 0)
    #libe1701py.mark_abs(cardNum, coord, coord, 0)
    
    libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
    time.sleep(0.2)
   # libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_STREAM, "0")

    libe1701py.execute(cardNum)

    time.sleep(5)
    
    libe1701py.close(cardNum)
else:
    print("Connection failed.")
