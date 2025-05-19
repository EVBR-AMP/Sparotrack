"""
Optimized optical-flow preview + UART velocity streaming
for Raspberry Pi 3B + Camera Module 2 using Picamera2.
"""

import cv2 as cv
import numpy as np
import time, serial, sys, signal
from picamera2 import Picamera2, Preview

# ---------- user settings ----------
SERIAL_PORT   = "/dev/serial0"
BAUD_RATE     = 115_200
RESOLUTION    = (160, 120)         # Lower resolution for Pi 3B
MAX_CORNERS   = 80                 # Fewer points to track
REFRESH_EVERY = 60                 # More frequent reseeds
SHOW_PREVIEW  = True              # Set to True to use cv.imshow
WINDOW_NAME   = "PiCam2 Optical Flow"

# ---------- graceful exit ----------
def stop(sig=None, frame=None):
    print("\nStopping…")
    picam2.stop()
    if SHOW_PREVIEW:
        cv.destroyAllWindows()
    ser.close()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)

# ---------- serial port ----------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ---------- camera ----------
picam2 = Picamera2()
video_cfg = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "RGB888"})
picam2.configure(video_cfg)
if not SHOW_PREVIEW:
    picam2.start_preview(Preview.DRM)  # Non-GUI preview
picam2.start()

# ---------- initial frame & features ----------
prev = picam2.capture_array()
prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)
p0 = cv.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 5)
prev_time = time.perf_counter()
frame_idx = 0

if SHOW_PREVIEW:
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

while True:
    frame = picam2.capture_array()
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # Optical flow
    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None:
        continue
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    flow = good_new - good_old

    if len(flow):
        dx, dy = np.mean(flow, axis=0)
        now = time.perf_counter()
        dt = now - prev_time
        if dt > 0:
            vy, vx = dx / dt, dy / dt
            ser.write(f"{vx:.2f},{vy:.2f}\n".encode())

            if SHOW_PREVIEW:
                cv.putText(frame, f"vx: {vx:6.2f} px/s   vy: {vy:6.2f} px/s",
                           (5, 20), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1, cv.LINE_AA)
                for pt in good_new.astype(int):
                    cv.circle(frame, tuple(pt), 1, (0, 0, 255), -1)
        prev_time = now

    # Reseed corners
    frame_idx += 1
    if frame_idx % REFRESH_EVERY == 0 or len(good_new) < MAX_CORNERS * 0.3:
        p0 = cv.goodFeaturesToTrack(gray, MAX_CORNERS, 0.01, 5)
        frame_idx = 0
    else:
        p0 = good_new.reshape(-1, 1, 2)

    prev_gray = gray

    # Optional preview
    if SHOW_PREVIEW:
        cv.imshow(WINDOW_NAME, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) & 0xFF == ord('q'):
            stop()
