from picamera2 import Picamera2, Preview
import cv2 as cv
import numpy as np
import time, signal, sys, serial

# ------------------------- settings -------------------------
LORES = (768, 432)          # (w, h)   16:9
MAX_CORNERS = 80
REFRESH_EVERY = 60
SHUTTER_US = 8000           # < 8333 µs for 120 fps
GAIN = 4.0                  # raise if image dark

SERIAL_PORT = "/dev/ttyAMA1"
BAUD = 921600
# ------------------------------------------------------------

# ---------- Serial setup ----------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
except serial.SerialException as e:
    print(f"Serial open failed: {e}")
    ser = None

# ---------- Camera setup ----------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    sensor={"output_size": (1536, 864), "bit_depth": 10},
    lores={"size": LORES, "format": "YUV420"},
    display="lores",
    buffer_count=6
)
picam2.align_configuration(cfg)
picam2.configure(cfg)
picam2.set_controls({
    "FrameDurationLimits": (8333, 8333),
    "ExposureTime": SHUTTER_US,
    "AnalogueGain": GAIN,
    "AeEnable": False
})
picam2.start_preview(Preview.NULL)
print("Sensor mode:", picam2.camera_config["sensor"])

# ---------- Globals ----------
prev_gray = None
t_prev = time.perf_counter()
frame_i = 0

# ---------- Callback ----------
def flow_cb(req):
    global prev_gray, t_prev, frame_i

    t_now = time.perf_counter()
    dt = t_now - t_prev
    t_prev = t_now
    if dt == 0:
        return

    gray = req.make_array("lores")[:LORES[1], :LORES[0]]
    if prev_gray is None:
        prev_gray = gray
        return

    p0 = cv.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 5)
    if p0 is not None:
        p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        if p1 is not None:
            flow = p1[st == 1] - p0[st == 1]
            if flow.size:
                dx, dy = np.mean(flow, axis=0)
                vx, vy = dx / dt, dy / dt
                print(f"vx:{vx:7.2f}  vy:{vy:7.2f}")

                # UART output
                if ser and ser.writable():
                    payload = f"{vx:.2f},{vy:.2f}\n"
                    try:
                        ser.write(payload.encode('utf-8'))
                    except serial.SerialException as e:
                        print(f"UART error: {e}")

    # Corner reseeding
    frame_i += 1
    if frame_i % REFRESH_EVERY == 0:
        prev_gray, frame_i = gray, 0
    else:
        prev_gray = gray

# ---------- Exit ----------
def stop(sig=None, frm=None):
    print("\nStopping …")
    picam2.stop()
    if ser:
        ser.close()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

# ---------- Start ----------
picam2.pre_callback = flow_cb
picam2.start()
signal.pause()
