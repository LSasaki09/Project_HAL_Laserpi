import serial
import struct
import time

class E1701RawController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(0.5)  # Let the controller initialize

    def close(self):
        self.ser.close()

    def flush(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_ascii_cmd(self, cmd: str):
        """Send a raw ASCII command like 'cslgt 1'."""
        full_cmd = (cmd.strip() + "\r\n").encode()
        self.ser.write(full_cmd)
        resp = self.ser.readline().decode().strip()
        return resp

    def send_d_command(self, cmd_type: int, x: int, y: int, z: int = 0):
        """
        Build and send a 14-byte 'd' command.

        cmd_type:
            0x01 = jump (laser off)
            0x02 = mark (laser on)
        x, y, z: 32-bit signed integers (usually range ±2^25)
        """
        payload = struct.pack("<cBiii", b'd', cmd_type, x, y, z)
        self.ser.write(payload)

        line = self.ser.readline()
        try:
            print("ASCII:", line.decode().strip())
        except:
            print("Raw (hex):", line.hex())


    def execute(self):
        """Flush the command buffer and execute."""
        self.send_ascii_cmd("csex")

    def laser_on(self):
        self.send_ascii_cmd("cslgt 1")

    def laser_off(self):
        self.send_ascii_cmd("cslgt 0")

    def set_laser_mode(self, on: bool):
        self.send_ascii_cmd(f"cslmo {1 if on else 0}")



    def set_laser_timing(self, freq=20000.0, pulse=50.0):
        self.send_ascii_cmd(f"csltf {freq:.1f} {pulse:.1f}")

    def set_speeds(self, jump_speed, mark_speed):
        self.send_ascii_cmd(f"csspd {jump_speed:.1f} {mark_speed:.1f}")

    def reset_position(self):
        self.send_d_command(0x01, 0, 0, 0)  # jump to (0,0)


if __name__ == "__main__":
    laser = E1701RawController("/dev/ttyACM0")

    x0, y0, z0 = 0, 0, 0
    square_size = 100000#2328550
    #jump_speed = 2000.0
    #mark_speed = 5000.0

    half = square_size // 2
    minx, maxx = x0 - half, x0 + half
    miny, maxy = y0 - half, y0 + half

    try:
        laser.set_laser_mode(1)
        laser.laser_off()
        #laser.set_laser_timing(20000.0, 50.0)
        #laser.set_speeds(jump_speed, mark_speed)

        # Jump to start
        laser.send_d_command(0x01, minx, miny)

        # Laser ON
        laser.laser_on()

        # Draw square
        laser.send_d_command(0x02, minx, maxy)
        laser.send_d_command(0x02, maxx, maxy)
        laser.send_d_command(0x02, maxx, miny)
        laser.send_d_command(0x02, minx, miny)

        # Return to (0,0)
        laser.send_d_command(0x02, 0, 0)

        # Laser OFF
        laser.laser_off()

        # Execute the sequence
        laser.execute()

        time.sleep(2)

    finally:
        laser.close()