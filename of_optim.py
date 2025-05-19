from picamera2 import Picamera2, Preview
import cv2 as cv
import numpy as np
import time
import signal
import sys

# ---------- Settings ----------
RESOLUTION = (224, 160)  # Small for optical flow
MAX_CORNERS = 80         # Fewer corners for speed
REFRESH_EVERY = 60       # Reseed every N frames

# ---------- Camera Setup ----------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    lores={'size': RESOLUTION, 'format': 'YUV420'},        # for optical flow
    main={'size': (640, 480), 'format': 'YUV420'},         # for preview
    display='main',
    buffer_count=4
)
picam2.configure(config)
picam2.start_preview(Preview.QTGL)  # Use Pi 5 GPU preview

# ---------- Globals ----------
prev_gray = None
prev_time = time.perf_counter()
frame_idx = 0

# ---------- Callback ----------
def flow_cb(request):
    global prev_gray, prev_time, frame_idx

    # Optical flow input (lores Y plane)
    yuv_lores = request.make_array("lores")
    gray = yuv_lores[:RESOLUTION[1], :RESOLUTION[0]]

    # Preview display frame (YUV420 to RGB)
    yuv_main = request.make_array("main")
    frame = cv.cvtColor(yuv_main, cv.COLOR_YUV2RGB_I420)

    now = time.perf_counter()

    if prev_gray is None:
        prev_gray = gray
        prev_time = now
        return

    p0 = cv.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 5)
    if p0 is None:
        prev_gray = gray
        return

    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None:
        prev_gray = gray
        return

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    flow = good_new - good_old

    if flow.size:
        dx, dy = np.mean(flow, axis=0)
        dt = now - prev_time or 1e-3
        vx, vy = dx / dt, dy / dt

        # Overlay
        cv.putText(frame, f"vx: {vx:.2f} px/s   vy: {vy:.2f} px/s",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for pt in good_new.astype(int):
            cv.circle(frame, tuple(pt), 2, (255, 0, 0), -1)

    # Reseed points if needed
    frame_idx += 1
    if frame_idx % REFRESH_EVERY == 0 or len(good_new) < MAX_CORNERS * 0.3:
        prev_gray = gray
        frame_idx = 0
    else:
        prev_gray = gray

    prev_time = now

    # Show frame
    cv.imshow("Flow Preview", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        stop()

# ---------- Graceful exit ----------
def stop(sig=None, frame=None):
    print("\nStopping…")
    picam2.stop()
    cv.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

# ---------- Run ----------
picam2.pre_callback = flow_cb
picam2.start()
signal.pause()
