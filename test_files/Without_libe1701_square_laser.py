import serial
import struct
import time

class E1701RawController:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)  # Suppression de newline='\r'
        time.sleep(0.5)  # Laisser le contrôleur s'initialiser

    def close(self):
        self.ser.close()

    def flush(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_ascii_cmd(self, cmd: str):
        """Envoyer une commande ASCII brute comme 'cvers' ou 'cslgt 1'."""
        full_cmd = (cmd.strip() + "\r\n").encode()
        self.ser.write(full_cmd)
        time.sleep(0.1)  # Petit délai pour la réponse
        line = self.ser.read_until(b'\r\n')  # Lire jusqu'à \r\n
        print(f"Raw response: {line}")
        try:
            resp = line.decode('latin1').strip()
        except UnicodeDecodeError:
            resp = f"Raw (hex): {line.hex()}"
        print(f"Decoded response: {resp}")
        return resp

    def send_d_command(self, cmd_type: int, x: int, y: int, z: int = 0):
        """
        Construire et envoyer une commande 'd' de 14 octets.
        cmd_type: 0x01 = jump (laser éteint), 0x02 = mark (laser allumé)
        x, y, z: Entiers signés 32 bits (généralement ±2^25)
        """
        payload = struct.pack("<cBiii", b'd', cmd_type, x, y, z)
        self.ser.write(payload)
        time.sleep(0.01)  # Petit délai après la commande

    def laser_on(self):
        self.send_ascii_cmd("cslgt 1")

    def laser_off(self):
        self.send_ascii_cmd("cslgt 0")

    def set_laser_timing(self, freq=20000.0, pulse=50.0):
        self.send_ascii_cmd(f"csltf {freq:.1f} {pulse:.1f}")

    def set_speeds(self, jump_speed, mark_speed):
        self.send_ascii_cmd(f"csspd {jump_speed:.1f} {mark_speed:.1f}")

    def reset_position(self):
        self.send_d_command(0x01, 0, 0, 0)  # Aller à (0,0,0)

    def get_version(self):
        """Obtenir la version du firmware avec 'cvers'."""
        return self.send_ascii_cmd("cvers")

if __name__ == "__main__":
    laser = E1701RawController("/dev/ttyACM0")

    x0, y0, z0 = 0, 0, 0
    square_size = 100000  # Ajuster selon la plage valide du scanner
    jump_speed = 2000.0
    mark_speed = 5000.0

    half = square_size // 2
    minx, maxx = x0 - half, x0 + half
    miny, maxy = y0 - half, y0 + half

    try:
        # Afficher la version du firmware
        print("Version du firmware :", laser.get_version())

        # Configurer le laser
        laser.laser_off()
        laser.set_laser_timing(20000.0, 50.0)
        laser.set_speeds(jump_speed, mark_speed)

        # Aller au point de départ
        laser.send_d_command(0x01, minx, miny)

        # Allumer le laser
        laser.laser_on()

        time.sleep(2)  # Attendre que le laser soit prêt

        # Dessiner un carré
        #laser.send_d_command(0x02, minx, maxy)
        #laser.send_d_command(0x02, maxx, maxy)
        #laser.send_d_command(0x02, maxx, miny)
        #laser.send_d_command(0x02, minx, miny)

        # Retourner à (0,0)
        laser.send_d_command(0x02, 0, 0)

        # Éteindre le laser
        laser.laser_off()

        time.sleep(2)  # Attendre la fin du marquage

        print("Marquage terminé avec succès")

    finally:
        laser.close()