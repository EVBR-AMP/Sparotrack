from picamera2 import Picamera2, Preview
import cv2 as cv, numpy as np, time, signal

RESOLUTION = (224, 160)
picam2 = Picamera2()
config = picam2.create_video_configuration(
    lores={'size': RESOLUTION, 'format': 'YUV420'},
    main={'size': (640, 480), 'format': 'RGB565'},
    display='main',
    buffer_count=4
)
picam2.configure(config)
picam2.start_preview(Preview.QTGL)  # Use Pi 5 GPU
picam2.start()

prev_gray = None
prev_t = time.perf_counter()

def flow_cb(req):
    global prev_gray, prev_t

    frame = req.make_array("main")  # For preview
    gray = req.make_array("lores")[:RESOLUTION[1], :RESOLUTION[0]]
    now = time.perf_counter()

    if prev_gray is None:
        prev_gray, prev_t = gray, now
        return

    p0 = cv.goodFeaturesToTrack(prev_gray, 80, 0.01, 5)
    if p0 is None:
        return
    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None:
        return

    flow = p1[st == 1] - p0[st == 1]
    if flow.size:
        dx, dy = np.mean(flow, axis=0)
        dt = now - prev_t or 1e-3
        vx, vy = dx / dt, dy / dt

        # Show debug overlay
        cv.putText(frame, f"vx: {vx:.2f}  vy: {vy:.2f}", (5, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show window
    cv.imshow("Flow Preview", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        picam2.stop()
        cv.destroyAllWindows()
        signal.raise_signal(signal.SIGINT)

    prev_gray, prev_t = gray, now

picam2.pre_callback = flow_cb
signal.pause()
