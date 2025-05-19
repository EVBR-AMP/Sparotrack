#!/usr/bin/env python3
"""
Show a live camera preview with optical-flow speed overlay and
stream vx, vy over UART0 (/dev/serial0) @ 115200 baud.
Tested on Raspberry Pi 5 + Camera Module 3 Wide + Picamera2 0.4+
"""

import cv2 as cv
import numpy as np
import time, serial, sys, signal
from picamera2 import Picamera2

# ---------- user settings ----------
SERIAL_PORT   = "/dev/serial0"     # alias for ttyAMA0 when Bluetooth is disabled
BAUD_RATE     = 115_200
RESOLUTION    = (320, 240)         # raise if you have spare CPU
MAX_CORNERS   = 120
REFRESH_EVERY = 90                 # frames between corner reseeds
WINDOW_NAME   = "PiCam2 Optical Flow (press q to quit)"

# ---------- graceful exit ----------
def stop(sig=None, frame=None):
    print("\nStopping…")
    picam2.stop()
    cv.destroyAllWindows()
    ser.close()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)

# ---------- serial port ----------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ---------- camera ----------
picam2 = Picamera2()
video_cfg = picam2.create_video_configuration(
        main={"size": RESOLUTION, "format": "RGB888"})  # OpenCV friendly
picam2.configure(video_cfg)
picam2.start()

# ---------- initial frame & features ----------
prev      = picam2.capture_array()
prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)
p0        = cv.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 7)

prev_time = time.perf_counter()
frame_idx = 0

cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

while True:
    frame = picam2.capture_array()
    gray  = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # — Lucas-Kanade sparse optical flow —
    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None:
        continue
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # velocity in px s-1
    flow = good_new - good_old
    if len(flow):
        dx, dy   = np.mean(flow, axis=0)
        now      = time.perf_counter()
        dt       = now - prev_time
        if dt > 0:
            vx, vy = dx / dt, dy / dt
            # send over UART
            ser.write(f"{vx:.2f},{vy:.2f}\n".encode())
            # overlay on frame
            cv.putText(frame, f"vx: {vx:6.2f} px/s   vy: {vy:6.2f} px/s",
                       (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2, cv.LINE_AA)
            # draw tracked features (optional eye-candy)
            for pt in good_new.astype(int):
                cv.circle(frame, tuple(pt), 2, (0, 0, 255), -1)
        prev_time = now

    # reseed corners periodically
    frame_idx += 1
    if frame_idx % REFRESH_EVERY == 0 or len(good_new) < MAX_CORNERS * 0.3:
        p0 = cv.goodFeaturesToTrack(gray, MAX_CORNERS, 0.01, 7)
        frame_idx = 0
    else:
        p0 = good_new.reshape(-1, 1, 2)

    prev_gray = gray

    # —— preview window ——
    cv.imshow(WINDOW_NAME, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    if cv.waitKey(1) & 0xFF == ord('q'):
        stop()
