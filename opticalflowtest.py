"""
Sparse Optical Flow demo (Lucas–Kanade) + average velocity
----------------------------------------------------------
Dependencies:
    pip install opencv-python numpy
Usage:
    • Press ESC to quit.
    • Change `0` in VideoCapture(0) to a file path if you want to process a video instead of the webcam.
"""

import cv2
import numpy as np

# ----- 1.  Video source -------------------------------------------------------
cap = cv2.VideoCapture(0)          # 0 = first webcam; or e.g. "myvideo.mp4"

if not cap.isOpened():
    raise IOError("Cannot open video source")

# ----- 2.  Parameters ---------------------------------------------------------
feature_params = dict(maxCorners=500,    # how many points to look for
                      qualityLevel=0.3,  # reject weak corners
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),       # LK search window
                 maxLevel=2,             # pyramid levels
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ----- 3.  Initialise with first frame & detect corners -----------------------
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read initial frame")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)          # for drawing tracks
rng_colors = np.random.randint(0, 255, (p0.shape[0], 3))

# ----- 4.  Main loop ----------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or read error"); break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track the previously detected points to the new frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # If tracking failed for every point, refresh the point set
    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask[:] = 0
        old_gray = frame_gray.copy()
        continue

    # Keep only successfully tracked points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), rng_colors[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 3, rng_colors[i].tolist(), -1)

    output = cv2.add(frame, mask)

    # ----- 5.  Compute & show average flow ------------------------------------
    flow_vecs = good_new - good_old         # Δx, Δy for every point
    if flow_vecs.size:
        avg_dx = float(np.mean(flow_vecs[:, 0]))
        avg_dy = float(np.mean(flow_vecs[:, 1]))
        text = f"Avg vx: {avg_dx:+.2f} px  |  Avg vy: {avg_dy:+.2f} px"
        cv2.putText(output, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # print to console as well
        print(f"\r{ text }", end="")

    cv2.imshow("Sparse Optical Flow (ESC to quit)", output)

    if cv2.waitKey(1) & 0xFF == 27:        # ESC key
        break

    # Update state for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# ----- 6.  Cleanup ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
