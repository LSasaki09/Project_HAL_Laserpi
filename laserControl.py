import time
import libe1701py
import serial.tools.list_ports

MAX_SPEED = 4294960000
MAX_POS = 67108860  # 2^26, for 26-bit precision
#freq = 10000 #20000.0  # Hz, laser frequency

def test_connection():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"{port.device} - {port.description}")

    cardNum = libe1701py.set_connection("/dev/ttyACM0")
    print(f"Cardnum: {cardNum}")

    ret = libe1701py.load_correction(cardNum, "", 0)  # set correction file, for no/neutral correction use "" or NULL here
    print(f"Correction data: {ret}")

    time.sleep(0.1)

    info = libe1701py.get_card_info(cardNum)
    print(f"Info: {info}")

    state = libe1701py.get_card_state(cardNum)
    print(f"State: {state}")
    
    time.sleep(0.5)
    return cardNum

def init_laser(port_name= "/dev/ttyACM0", freq = 10000, jump_speed = MAX_SPEED//100, mark_speed = 50000,
                corr_file="corr_file_v4.bco"):

    """Initialize the laser with the given parameters."""
    # ==== Setup scanner ====
    cardNum = libe1701py.set_connection(port_name)
    ret = libe1701py.load_correction(cardNum, corr_file, 0)
    if ret == libe1701py.E170X_OK:
        libe1701py.tune(cardNum, libe1701py.E170X_TUNE_XY2_18BIT)  # for XY2E-100 protocol (18-bit precision)
        
        libe1701py.set_laser_mode(cardNum, libe1701py.E170X_LASERMODE_CO2)
        libe1701py.set_laser_timing(cardNum, freq, 50.0)  #25.0 for 50% duty cycle

        libe1701py.set_speeds(
            cardNum,
            jump_speed,
            mark_speed
        )  # Max speed for jump and mark
        time.sleep(0.05)
    else:
        print(f"ERROR: opening correction file failed with error {ret}")
    return cardNum



def wait_marking(cardNum):
    """Wait until the marking is done."""
    state = libe1701py.get_card_state(cardNum)
    while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) == 0:
        time.sleep(0.01)
        state = libe1701py.get_card_state(cardNum)

    # Wait for marking to complete
    while (state & (libe1701py.E170X_CSTATE_MARKING | libe1701py.E170X_CSTATE_PROCESSING)) != 0:
        time.sleep(0.05)
        state = libe1701py.get_card_state(cardNum)



def draw_square(cardNum, square_size, x0=0, y0=0, z=0):
    """Draw a square around the central point."""
    half_size = square_size // 2
    libe1701py.mark_abs(cardNum, x0 - half_size, y0 - half_size, z)  # Lower left corner
    libe1701py.mark_abs(cardNum, x0 - half_size, y0 + half_size, z)  # Upper left corner
    libe1701py.mark_abs(cardNum, x0 + half_size, y0 + half_size, z)  # Upper right corner
    libe1701py.mark_abs(cardNum, x0 + half_size, y0 - half_size, z)  # Lower right corner
    libe1701py.mark_abs(cardNum, x0 - half_size, y0 - half_size, z)  # Back to the starting corner
    libe1701py.mark_abs(cardNum, x0, y0, z)  # Return to origin
    libe1701py.execute(cardNum)

    time.sleep(0.2)
    wait_marking(cardNum)
    time.sleep(0.1)
    libe1701py.close(cardNum)



def draw_grid_pts(cardNum, grid_size=3 , bit_range = 67108860//2):
    """Draw a grid of points."""
    # ==== Define scan grid ====
    bit_positions = np.linspace(-bit_range, bit_range, grid_size, dtype=int)
    grid_points = [(x, y) for x in bit_positions for y in bit_positions]

    for idx, (bx, by) in enumerate(grid_points):
        print(f"[{idx+1}/{len(grid_points)}] Point: ({bx}, {by})")

        # Move laser to position
        libe1701py.jump_abs(cardNum, bx, by, 0)
        libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "1")
        libe1701py.execute(cardNum)
        time.sleep(0.3)

        # Turn off laser
        #libe1701py.set_laser(cardNum, libe1701py.E170X_COMMAND_FLAG_DIRECT, "0")
        #libe1701py.execute(cardNum)
        #time.sleep(laser_on_duration)

        wait_marking(cardNum)

        print(f"Point ({bx}, {by}) processed.")
        time.sleep(0.1)

    libe1701py.close(cardNum)






