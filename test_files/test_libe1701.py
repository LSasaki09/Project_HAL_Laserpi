import libe1701py
import time
import numpy as np

# Callback function to set power for a CO2 laser
def set_power(n, power, userData):
    """
    Set power for a CO2 laser in unit % using frequency of 20 kHz and maximum duty cycle of 50%.
    HANDLE WITH CARE!!! This function is called every time the power changes from one pixel to
    another. Complex calculations here can cause a big load on the host computer, and lots of
    E170X-functions called here can cause a lot of load on Ethernet/USB connection and on the
    controller itself!
    """
    maxDutyCycle = 0.5
    halfPeriod = int(((1.0 / 20000.0) / 2.0) * 1000000.0)  # usec
    if halfPeriod < 2:
        halfPeriod = 2
    elif halfPeriod > 65500:
        halfPeriod = 65500
    pwrPeriod1 = int(halfPeriod * 2 * (power / 100.0) * maxDutyCycle)
    return libe1701py.set_laser_timing(n, 20000, pwrPeriod1)

# Main execution
cardNum = libe1701py.set_connection("/dev/ttyACM1") # "COM3" on Windows
if cardNum > 0:
    ret = libe1701py.load_correction(cardNum, "", 0)  # Load correction file, use "" for no correction
    if ret == libe1701py.E170X_OK:
        # Configure the scanhead
        libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY3_20BIT)  # Enable XY3-100 control mode

        # Set general marking parameters
        libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)  # Configure for CO2
        libe1701py.set_standby2(cardNum, 20000.0, 5.0, False)  # 20kHz and 5 usec standby frequency/period
        libe1701py.set_laser_timing(cardNum, 20000.0, 25.0)  # 20kHz and 50% duty cycle marking frequency/period
        libe1701py.set_scanner_delays2(cardNum, 0, 100.0, 100.0, 10.0)  # Delays, adjust for scanhead
        libe1701py.set_laser_delays(cardNum, 20.0, 30.0)  # Laser on/off delays in usec
        libe1701py.set_speeds(cardNum, 67108864.0, 67108.864)  # Jump and mark speeds in bits/ms
        libe1701py.digi_set_motf(cardNum, 0.0, 0.0)  # No marking on-the-fly

        # Send data to perform a normal mark operation with vector data
        libe1701py.jump_abs(cardNum, -10000000, -10000000, -10000000)  # Jump to start position
        libe1701py.mark_abs(cardNum, -10000000, 10000000, 10000000)  # Mark a square
        libe1701py.mark_abs(cardNum, 10000000, 10000000, -10000000)
        libe1701py.mark_abs(cardNum, 10000000, -10000000, 10000000)
        libe1701py.mark_abs(cardNum, -10000000, -10000000, -10000000)

        # Send data to mark a single pixel line
        NUM_PIXELS = 500
        pixelLine_list = [i / 5.0 for i in range(NUM_PIXELS)]  # Liste Python
        pixelLine = np.array(pixelLine_list, dtype=np.float64)  # Conversion en tableau NumPy float64

        # Configurer les paramètres du laser et du scanner
        libe1701py.set_laser_delays(cardNum, 0.0, 0.5)  # Petits délais pour le marquage bitmap
        libe1701py.set_scanner_delays2(cardNum, 0, 0.0, 0.0, 0.0)
        libe1701py.set_pixelmode(cardNum, 0, 2.0, 0)  # Mode de marquage rapide avec sauts lorsque le laser est éteint

        # Appel de mark_pixelline avec 9 arguments
        libe1701py.mark_pixelline(cardNum, 0, 0, 0, 10, 0, 0, NUM_PIXELS, pixelLine)
        # Ensure marking operation is started
        libe1701py.execute(cardNum)

        # Wait until marking has started
        state = libe1701py.get_card_state(cardNum)
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) == 0:
            time.sleep(0.01)
            state = libe1701py.get_card_state(cardNum)

        # Wait until marking is finished
        while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) != 0:
            time.sleep(0.5)
            state = libe1701py.get_card_state(cardNum)
            

        # Get the last position
        ret, x, y, z = libe1701py.get_pos(cardNum)

        if ret == libe1701py.E170X_OK:
            print(f"Last position: x={x}, y={y}, z={z}")
        else:
            print(f"ERROR: get_pos failed with error {ret}")

        time.sleep(1)
        libe1701py.close(cardNum)
    else:
        print(f"ERROR: opening connection failed with error {ret}")
else:
    print("ERROR: Could not initialise!")