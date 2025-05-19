"""
Streams optical-flow velocities (vx, vy) over UART0 (/dev/ttyAMA0) at 115200 baud
using Picamera2 and Camera Module 3 Wide on Raspberry Pi 5.
"""
import cv2 as cv
import numpy as np
import time, serial, sys, signal
from picamera2 import Picamera2

# ---------- user settings ----------
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115_200
RESOLUTION  = (320, 240)   # tune for speed vs. accuracy
MAX_CORNERS = 120
REFRESH_EVERY = 90         # reseed corners every N frames

# ---------- graceful exit ----------
def stop(sig, frame):
    print("\nStopping…")
    picam2.stop()
    ser.close()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)

# ---------- serial port ----------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ---------- camera ----------
picam2 = Picamera2()
video_cfg = picam2.create_video_configuration(
        main={"size": RESOLUTION, "format": "RGB888"})   # OpenCV-friendly format :contentReference[oaicite:6]{index=6}
picam2.configure(video_cfg)
# continuous autofocus is the default; leave it running
picam2.start()

# ---------- initial frame & features ----------
prev = picam2.capture_array()
prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)
p0 = cv.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 7)

prev_time = time.perf_counter()
frame_idx = 0

while True:
    frame = picam2.capture_array()
    gray  = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None:
        continue
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # mean displacement → velocity in px s-1
    flow = good_new - good_old
    if len(flow):
        dx, dy = np.mean(flow, axis=0)
        now = time.perf_counter()
        dt  = now - prev_time
        if dt > 0:
            vx, vy = dx / dt, dy / dt
            ser.write(f"{vx:.2f},{vy:.2f}\n".encode())
        prev_time = now

    # refresh feature set
    frame_idx += 1
    if frame_idx % REFRESH_EVERY == 0 or len(good_new) < MAX_CORNERS*0.3:
        p0 = cv.goodFeaturesToTrack(gray, MAX_CORNERS, 0.01, 7)
        frame_idx = 0
    else:
        p0 = good_new.reshape(-1, 1, 2)

    prev_gray = gray
