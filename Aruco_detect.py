import cv2
import numpy as np
import yaml
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
    dist_coeffs   = np.asarray(calib["dist_coeff"],   dtype=np.float32).ravel()
    print("Loaded camera intrinsics from", CALIB_FILE)
except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
    sys.exit(f"❌  Could not read {CALIB_FILE}: {e}")

# ---------------------------------------------------------------------
# 2. User-configurable parameters
# ---------------------------------------------------------------------
SERIAL_PORT = '/dev/ttyAMA0'   # UART device
BAUD_RATE   = 115200           # Baud rate

# Marker geometry (metres)
marker_length = 0.10                     # 10 cm square
half = marker_length / 2.0

# Board layout (origin at midpoint between markers)
# Centers in METRES in the board frame:
# left id=0 at (0, -0.0725), right id=1 at (0, +0.0725)
c0 = np.array([-0.07, 0.12, 0.0], dtype=np.float32)
c1 = np.array([0.07,  0.12, 0.0], dtype=np.float32)

def corners_from_center(center):
    cx, cy, cz = center
    return np.array([
        [cx - half, cy - half, cz],
        [cx + half, cy - half, cz],
        [cx + half, cy + half, cz],
        [cx - half, cy + half, cz],
    ], dtype=np.float32)

# Per-marker corner model for SINGLE-marker PnP (origin at the marker center)
single_marker_points = np.array(
    [
        [-half, -half, 0.0],
        [ half, -half, 0.0],
        [ half,  half, 0.0],
        [-half,  half, 0.0],
    ],
    dtype=np.float32,
)

# Offsets from each marker center to the BOARD origin (metres)
offset_to_board_from = {
    0: np.array([0.0,  0.0, 0.0], dtype=np.float32),  # from id 0 center to (0,0)
    1: np.array([0.0,  0.0, 0.0], dtype=np.float32),  # from id 1 center to (0,0)
}

# ---------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------
def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    # ZYX (yaw-pitch-roll), degrees
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
# 4. ArUco detector + BOARD definition (origin at midpoint)
# ---------------------------------------------------------------------
dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

params  = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.04
params.maxMarkerPerimeterRate = 4.0
params.minMarkerDistanceRate   = 0.05
params.minCornerDistanceRate   = 0.05
params.minDistanceToBorder     = 3
params.adaptiveThreshWinSizeMin  = 15
params.adaptiveThreshWinSizeMax  = 55
params.adaptiveThreshWinSizeStep = 10
params.adaptiveThreshConstant    = 7

detector = cv2.aruco.ArucoDetector(dictionary, params)

# Custom board with origin at (0,0,0) midway between the markers
objPoints = [corners_from_center(c0), corners_from_center(c1)]
ids_board = np.array([0, 1], dtype=np.int32)

# OpenCV has two APIs depending on version; try both.
try:
    board = cv2.aruco.Board(objPoints, dictionary, ids_board)
except Exception:
    board = cv2.aruco.Board_create(objPoints, dictionary, ids_board)

# ---------------------------------------------------------------------
# 5. Initialise camera and serial port
# ---------------------------------------------------------------------
try:
    FPS = 56
    FRAME_DURATION = int(1_000_000 / FPS)  # µs

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        sensor={'output_size': (2304, 1296)},
        main={'format': 'BGR888', 'size': (1152, 648)}
    )
    picam2.configure(cfg)
    picam2.start()

    picam2.set_controls({
        "FrameDurationLimits": (FRAME_DURATION, FRAME_DURATION),
        "AeEnable": False,
        "ExposureTime": 4000,     # µs
        "AnalogueGain": 6.7,
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

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Optional refinement using board geometry
    if True:
        try:
            cv2.aruco.refineDetectedMarkers(
                image=frame,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
        except Exception:
            pass

    pose_texts = []
    have_pose = False

    # --- Primary: fuse BOTH markers as a board (origin at midpoint) ---
    if ids is not None and len(ids) > 0:
        try:
            ok, rvec_board, tvec_board = cv2.aruco.estimatePoseBoard(
                corners, ids, board, camera_matrix, dist_coeffs, None, None
            )
        except Exception:
            ok = 0

        if ok and ok > 0:
            R, _ = cv2.Rodrigues(rvec_board)
            t_cm = tvec_board * 100.0
            x_cm = t_cm[0, 0]
            y_cm = t_cm[1, 0]
            yaw  = rotation_matrix_to_euler(R)[2]
            have_pose = True

            payload = f"{x_cm:.1f},{y_cm:.1f},{yaw:.1f}\n"
            try:
                ser.write(payload.encode('utf-8'))
            except serial.SerialException as e:
                print(f"Serial write error: {e}")

            # print(f"(0,0) pose  X={x_cm:.1f} cm  Y={y_cm:.1f} cm  Yaw={yaw:.1f}°")
            # pose_texts.append(f"Midpoint X[{x_cm:.1f}] cm  Y[{y_cm:.1f}] cm  Yaw[{yaw:.1f}]°")

            # # Draw axes at the board origin (midpoint)
            # try:
                # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_board, tvec_board, 0.1)
            # except Exception:
                # pass

    # --- Fallback: only one tag visible → shift to board origin ---
    if not have_pose and ids is not None:
        ids_list = ids.flatten().tolist()
        for idx, marker_corners in zip(ids_list, corners):
            if idx in (0, 1):
                success, rvec_m, tvec_m = cv2.solvePnP(
                    single_marker_points,
                    marker_corners[0],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE  # better for planar squares
                )
                if not success:
                    continue

                Rm, _ = cv2.Rodrigues(rvec_m)
                # Shift from marker center to board origin
                offset = offset_to_board_from[idx].reshape(3, 1)
                tvec_board = (Rm @ offset) + tvec_m
                rvec_board = rvec_m  # same rotation for the rigid body

                R, _ = cv2.Rodrigues(rvec_board)
                t_cm = tvec_board * 100.0
                x_cm = t_cm[0, 0]
                y_cm = t_cm[1, 0]
                yaw  = rotation_matrix_to_euler(R)[2]
                have_pose = True

                payload = f"{x_cm:.1f},{y_cm:.1f},{yaw:.1f}\n"
                try:
                    ser.write(payload.encode('utf-8'))
                except serial.SerialException as e:
                    print(f"Serial write error: {e}")

                # print(f"(0,0) (fallback via id {idx})  X={x_cm:.1f} cm  Y={y_cm:.1f} cm  Yaw={yaw:.1f}°")
                # pose_texts.append(f"Midpoint (id {idx}) X[{x_cm:.1f}] cm  Y[{y_cm:.1f}] cm  Yaw[{yaw:.1f}]°")

                # # Draw axes at the inferred board origin
                # try:
                    # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_board, tvec_board, 0.1)
                # except Exception:
                    # pass
                # break  # stop after first valid fallback

    # # Draw detections
    # if ids is not None and len(ids) > 0:
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # # OSD text
    # for i, text in enumerate(pose_texts):
        # cv2.putText(
            # frame, text, (10, 20 + 20 * i),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        # )

    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
        # break

# ---------------------------------------------------------------------
# 7. Cleanup
# ---------------------------------------------------------------------
picam2.stop()
ser.close()
cv2.destroyAllWindows()
print("Camera stopped, serial port closed.")
