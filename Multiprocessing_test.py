import cv2
import numpy as np
import time
import random
import threading
from multiprocessing import Process, Value, Event
import csv
from picamera2 import Picamera2
import libe1701py
from cv2 import aruco
import picamFunctions as pf
import laserControl as lc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import fsolve

def print_a_center(centers, i):
    time.sleep(2)
    n = len(centers)
    if i < n:
        print(centers[i])
    elif i == -1:
        print(f"There are {n} centers.")
    else:
        print(f"The center {i} doesn't exists !")

def compute_velocity(memory_vel, stop_event):
    while not stop_event.is_set():
        memory_vel.value = 20*(random.random()-0.5)
        time.sleep(1)

def compute_position(memory_vel, stop_event):
    dt = 0.01
    pos = 0.0
    start = time.perf_counter()

    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots()
    ax.set_xlabel("X")
    ax.set_ylabel("Time (s)")
    ax.set_title("Trajectory of points")
    ax.grid(True)

    x = []
    y = []
    while not stop_event.is_set():
        pos = pos + dt*memory_vel.value
        t = time.perf_counter() - start

        x.append(pos)
        y.append(t)

        ax.clear()  # clear the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Time (s)")
        ax.set_title("Trajectory of points")
        ax.grid(True)

        ax.scatter(x, y)
        plt.pause(0.01)  # let GUI update
        time.sleep(dt)


if __name__ == "__main__":

    picam2, m, d = pf.init_camera()
    time.sleep(0.5)
    cardNum = lc.init_laser()

    unit = "pixels"  # or "mm"

    if unit == "mm":
        corr_file="corr_file_v4.bco"
    elif unit == "pixels":
        corr_file="corr_file_v5.bco"
    else:
        print("ERROR: Invalid unit. Use 'mm' or 'pixels'.")
        exit(1)
    
    # Path to CSV file
    csv_path = "scanner_camera_map.csv"


    # Test for print_a_center()
    """
    frame = picam2.capture_array()
    centers, _, _, _ = pf.detect_aruco_markers(frame, mtx = m, dist = d)

    for i in range(4):
        print_a_center(centers, i)
        print(f"Finished printing the {i}th center !")

    print("Starting the multiprocessing procedure")
    procs = [Process(target = print_a_center, args = (centers, i)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("Wouah ! That was fast !")
    """

    # Test for compute_velocity / compute_position functions
    """
    stop_event = Event()
    m_vel = Value('d', 0.0)
    p1 = Process(target=compute_velocity, args=(m_vel, stop_event))
    p2 = Process(target=compute_position, args=(m_vel, stop_event))

    p1.start(), p2.start()

    time.sleep(20)
    stop_event.set()
    p1.join(), p2.join()
    """