import serial
import struct
import time
import numpy as np

class E1701RawController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        """Initialize serial connection to the E1701 controller."""
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(0.5)  # Allow controller to initialize
        self.flush()

    def close(self):
        """Close the serial connection."""
        self.ser.close()

    def flush(self):
        """Clear input and output buffers."""
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_ascii_cmd(self, cmd: str):
        """Send an ASCII command and handle echo."""
        full_cmd = (cmd.strip() + "\r\n").encode()
        self.ser.write(full_cmd)
        time.sleep(0.1)  # Wait for response
        # Read echo
        echo = self.ser.read_until(b'\r\n')
        print(f"Echo: {echo.decode('latin1').strip()}")
        # Read actual response
        line = self.ser.read_until(b'\r\n')
        print(f"Raw response: {line}")
        try:
            resp = line.decode('latin1').strip()
        except UnicodeDecodeError:
            resp = f"Raw (hex): {line.hex()}"
        print(f"Decoded response: {resp}")
        return resp

    def send_d_command(self, cmd_type: int, x: int, y: int, z: int = 0):
        """
        Send a 14-byte 'd' command.
        cmd_type: 0x01 = jump (laser off), 0x02 = mark (laser on)
        x, y, z: Signed 32-bit integers (typically ±2^25)
        """
        payload = struct.pack("<cBiii", b'd', cmd_type, x, y, z)
        self.ser.write(payload)
        time.sleep(0.05)  # Increased delay for reliability

    def laser_on(self):
        """Turn the laser on with 'cslgt 1'."""
        resp = self.send_ascii_cmd("cslgt 1")
        if not resp.startswith("OK"):
            print(f"Error: 'cslgt 1' returned '{resp}'")

    def laser_off(self):
        """Turn the laser off with 'cslgt 0'."""
        resp = self.send_ascii_cmd("cslgt 0")
        if not resp.startswith("OK"):
            print(f"Error: 'cslgt 0' returned '{resp}'")

    def set_laser_timing(self, freq=20000.0, pulse=50.0):
        """Set laser timing with 'csltf'."""
        resp = self.send_ascii_cmd(f"csltf {freq:.1f} {pulse:.1f}")
        if not resp.startswith("OK"):
            print(f"Error: 'csltf' returned '{resp}'")

    def set_speeds(self, jump_speed, mark_speed):
        """Set jump and mark speeds with 'csspeed'."""
        resp = self.send_ascii_cmd(f"csspeed {jump_speed:.1f} {mark_speed:.1f}")
        if not resp.startswith("OK"):
            print(f"Error: 'csspeed' returned '{resp}'")

    def reset_position(self):
        """Move to (0,0,0) with jump command."""
        self.send_d_command(0x01, 0, 0, 0)

    def get_version(self):
        """Get firmware version with 'cvers'."""
        return self.send_ascii_cmd("cvers")

    def init_controller(self):
        """Initialize controller with connection and correction settings."""
        self.send_ascii_cmd("csini")  # Initialize connection (assumed for set_connection)
        resp = self.send_ascii_cmd("cload 0")  # Load default correction table
        if not resp.startswith("OK"):
            print(f"Error: 'cload 0' returned '{resp}'. Check correction file.")
        self.send_ascii_cmd("csmod 0")  # Enable direct mode

    def get_internal_time(self):
        """Get controller's internal clock time in microseconds."""
        resp = self.send_ascii_cmd("cgtim")
        if resp.startswith("OK:"):
            try:
                return int(resp.split(":")[1].strip())
            except (IndexError, ValueError):
                print(f"Error: Invalid 'cgtim' response: {resp}")
                return None
        print(f"Error: 'cgtim' returned '{resp}'")
        return None

    def measure_laser_on_delay(self, num_measurements=100, timeout=1.0):
        """Measure delay between sending 'cslgt 1' and laser activation using controller state."""
        delays = []
        for i in range(num_measurements):
            # Ensure laser is off
            self.laser_off()
            time.sleep(0.1)  # Stabilize initial state

            # Send laser on command and get start time
            self.ser.write(b"cslgt 1\r\n")
            self.ser.read_until(b'\r\n')  # Read echo
            start_time = self.get_internal_time()
            if start_time is None:
                print(f"Warning: Failed to get start time in iteration {i+1}")
                continue
            resp = self.ser.read_until(b'\r\n').decode('latin1').strip()
            if not resp.startswith("OK"):
                print(f"Warning: 'cslgt 1' returned '{resp}' in iteration {i+1}")
                continue

            # Poll controller status until idle
            start_poll = time.perf_counter()
            while time.perf_counter() - start_poll < timeout:
                status = self.send_ascii_cmd("cgsta")
                end_time = self.get_internal_time()
                if end_time is None:
                    print(f"Warning: Failed to get end time in iteration {i+1}")
                    break
                if status == "0":  # Idle state indicates command completion
                    delay = (end_time - start_time) / 1_000_000  # Convert microseconds to seconds
                    delays.append(delay)
                    print(f"Iteration {i+1}: Laser ON completed, delay = {delay:.6f} seconds")
                    break
                time.sleep(0.001)  # Reduce CPU load
            else:
                print(f"Timeout: Laser ON not completed in iteration {i+1}")

        # Analyze delays
        if delays:
            delays_np = np.array(delays)
            print("\nLaser ON Delay Statistics:")
            print(f"  Mean: {np.mean(delays_np):.6f} seconds")
            print(f"  Standard Deviation: {np.std(delays_np):.6f} seconds")
            print(f"  Min: {np.min(delays_np):.6f} seconds")
            print(f"  Max: {np.max(delays_np):.6f} seconds")
            print(f"  Median: {np.median(delays_np):.6f} seconds")
        else:
            print("No valid delays recorded.")
        return delays

if __name__ == "__main__":
    laser = E1701RawController("/dev/ttyACM0")

    x0, y0, z0 = 0, 0, 0
    square_size = 10000  # Reduced for safety
    jump_speed = 2000.0
    mark_speed = 5000.0

    try:
        # Initialize controller
        laser.init_controller()

        # Display firmware version
        print("Firmware Version:", laser.get_version())

        # Configure laser
        laser.laser_off()
        laser.set_laser_timing(20000.0, 5000.0)
        laser.set_speeds(jump_speed, mark_speed)

        # Measure laser ON delay
        #delays = laser.measure_laser_on_delay()

        # Move to starting point
        #laser.reset_position()

        # Draw a square (uncomment to enable)
        # laser.laser_on()
        # time.sleep(2)  # Wait for laser to stabilize
        # half = square_size // 2
        # minx, maxx = x0 - half, x0 + half
        # miny, maxy = y0 - half, y0 + half
        # laser.send_d_command(0x02, minx, maxy)
        # laser.send_d_command(0x02, maxx, maxy)
        # laser.send_d_command(0x02, maxx, miny)
        # laser.send_d_command(0x02, minx, miny)
        # laser.send_d_command(0x02, 0, 0)
        # laser.laser_off()

        time.sleep(2)  # Wait for completion
        print("Operation completed successfully")

    finally:
        laser.close()