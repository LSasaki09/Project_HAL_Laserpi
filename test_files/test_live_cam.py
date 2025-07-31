import cv2
from picamera2 import Picamera2
import time
import sys
sys.path.append("/usr/lib/python3/dist-packages")

# Initialise la caméra
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"size": (640, 480)})) # 1280, 720
picam.start()

time.sleep(1)  # Laisse le capteur s'adapter

# Boucle d'affichage
print("🟢 Press 'q' to quit")
while True:
    frame = picam.capture_array()
    cv2.imshow("Live Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

picam.stop()
cv2.destroyAllWindows()
