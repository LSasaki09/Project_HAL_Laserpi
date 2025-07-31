'''This script is for generating data
1. Provide desired path to store images.
2. Press 'c' to capture image and display it.
3. Press any button to continue.
4. Press 'q' to quit.
'''
import cv2
from picamera2 import Picamera2

# Initialiser la caméra avec Picamera2
picam2 = Picamera2()
picam2.start()


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()


# Chemin pour sauvegarder les images
path = "/home/pi/projects/Project_HAL_Laserpi/cam_calibration/aruco_data/"
count = 0

while True:
    name = path + str(count) + ".jpg"
    # Capturer une image sous forme de tableau numpy
    img = picam2.capture_array()
    # Convertir de RGB (Picamera2) à BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)

    # Attendre la touche 'c' pour capturer
    if cv2.waitKey(20) & 0xFF == ord('c'):
        cv2.imwrite(name, img)
        print(f"Image sauvegardée : {name}")
        count += 1
        # Attendre 'q' pour quitter après capture
         key = cv2.waitKey(1) & 0xFF
         if key == ord("q"):
        break

# Nettoyage
picam2.stop()
cv2.destroyAllWindows()