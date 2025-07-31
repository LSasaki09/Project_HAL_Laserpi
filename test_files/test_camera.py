from picamera2 import Picamera2
import time
import sys
sys.path.append("/usr/lib/python3/dist-packages")

picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()
time.sleep(2)
picam.capture_file("test.jpg")
picam.close()
print("Image captured !")
