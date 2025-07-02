import cv2
import numpy as np
import yaml                             # ← NEW
from picamera2 import Picamera2
import serial
import time
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load camera intrinsics from camera.yaml
# ---------------------------------------------------------------------
CALIB_FILE = Path("camera.yaml")

try:
    with CALIB_FILE.open() as f:
        calib = yaml.safe_load(f)
    camera_matrix = np.asarray(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs   = np.asarray(calib["dist_coeff"],   dtype=np.float32)

    # OpenCV accepts 1×N or N×1; keep them flat for solvePnP
    dist_coeffs = dist_coeffs.ravel()

    print("Loaded camera intrinsics from", CALIB_FILE)
except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
    sys.exit(f"❌  Could not read {CALIB_FILE}: {e}")

# ---------------------------------------------------------------------
# 2. User-configurable parameters
# ---------------------------------------------------------------------
SERIAL_PORT = '/dev/ttyAMA0'   # UART device
BAUD_RATE   = 115200           # Baud rate

# Marker size in metres (e.g. 0.10 m for a 10 cm square)
marker_length = 0.1
marker_points = np.array(
    [
        [-marker_length / 2, -marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [-marker_length / 2,  marker_length / 2, 0],
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------
# 3. Helper: rotation matrix → Euler angles (ZYX) in degrees
# ---------------------------------------------------------------------
def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

# ---------------------------------------------------------------------
# 4. ArUco detector setup
# ---------------------------------------------------------------------
dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params  = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.04   # default 0.03
params.maxMarkerPerimeterRate = 4.0    # just in case you also see giant blobs
params.minMarkerDistanceRate   = 0.05  # reject overlapping blobs
params.minCornerDistanceRate   = 0.05
params.minDistanceToBorder     = 3     # px; 0 = accept at the very edge
params.adaptiveThreshWinSizeMin  = 15   # default 3
params.adaptiveThreshWinSizeMax  = 55   # default 23
params.adaptiveThreshWinSizeStep = 10   # default 10 is fine
params.adaptiveThreshConstant    = 7    # tweak ±2–3 until dots vanish

detector    = cv2.aruco.ArucoDetector(dictionary, params)

# ---------------------------------------------------------------------
# 5. Initialise camera and serial port
# ---------------------------------------------------------------------
try:
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (1456, 1088)}
        )
    )
    picam2.start()
    picam2.set_controls({
    "AeEnable": False,        # turn off auto exposure/gain
    "ExposureTime": 4000,     # µs
    "AnalogueGain": 6.7,      # ×
    })
except Exception as e:
    sys.exit(f"❌  Could not open camera: {e}")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial port {SERIAL_PORT} opened at {BAUD_RATE} baud")
except serial.SerialException as e:
    picam2.stop()
    sys.exit(f"❌  Error opening serial port: {e}")

print("Camera running — press ‘q’ to quit")

# ---------------------------------------------------------------------
# 6. Main loop
# ---------------------------------------------------------------------
while True:
    frame = picam2.capture_array()
    if frame is None:
        print("⚠️  Empty frame received, exiting.")
        break

    corners, ids, _ = detector.detectMarkers(frame)
    pose_texts = []

    if ids is not None:
        for marker_corners in corners:
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                marker_corners[0],
                camera_matrix,
                dist_coeffs,
            )
            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)

            t_cm = tvec * 100.0

            x_cm =  t_cm[0][0]
            y_cm =  t_cm[1][0]
            euler = rotation_matrix_to_euler(R)
            yaw  =  euler[2]

            payload = f"{x_cm:.1f},{y_cm:.1f},{yaw:.1f}\n"
            try:
                ser.write(payload.encode('utf-8'))
            except serial.SerialException as e:
                print(f"Serial write error: {e}")
            
            print(f"Marker at  X={x_cm:.1f} cm  Y={y_cm:.1f} cm  Yaw={yaw:.1f}°")

            pose_texts.append(
                f"X[{x_cm:.1f}] cm  Y[{y_cm:.1f}] cm  Yaw[{yaw:.1f}]°"
            )

            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
            )

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    for i, text in enumerate(pose_texts):
        cv2.putText(
            frame,
            text,
            (10, 20 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------------------------------------------------
# 7. Cleanup
# ---------------------------------------------------------------------
picam2.stop()
ser.close()
cv2.destroyAllWindows()
print("Camera stopped, serial port closed.")
