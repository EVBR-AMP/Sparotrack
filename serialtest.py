import cv2
import numpy as np
from picamera2 import Picamera2
import serial
import time

# Define the serial port and baud rate
SERIAL_PORT = '/dev/ttyAMA0'  # Explicitly use ttyAMA0 as the UART device
BAUD_RATE = 115200            # Common baud rate for testing

# Function to compute Euler angles from a rotation matrix (ZYX convention) in degrees
def rotation_matrix_to_euler(R):
    """Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # Roll
        y = np.arctan2(-R[2,0], sy)      # Pitch
        z = np.arctan2(R[1,0], R[0,0])   # Yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z]) * 180 / np.pi

# Camera parameters (placeholders for a 640x480 resolution)
fx, fy = 640.0, 640.0  # Focal lengths in pixels
cx, cy = 320.0, 240.0  # Principal point (image center)
k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0  # Distortion coefficients

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Define marker size in meters (e.g., 10 cm square marker)
marker_length = 0.1

# 3D points of the marker in its own coordinate system (Z=0 plane)
marker_points = np.array([
    [-marker_length / 2, -marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [-marker_length / 2,  marker_length / 2, 0]
], dtype=np.float32)

# Set up ArUco dictionary and detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Initialize Picamera2
try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
    picam2.start()
except Exception as e:
    print(f"Error: Could not open camera. {e}")
    exit()

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    picam2.stop()
    exit()

print("Camera started. Press 'q' to stop (if display is available).")

# Main loop
while True:
    # Capture frame from Picamera2
    frame = picam2.capture_array()
    if frame is None:
        print("Error: Could not read frame.")
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    pose_texts = []  # List to store pose information for each marker
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            # Get the 2D corners of the current marker
            marker_corners = corners[i][0]

            # Estimate pose using solvePnP (marker pose in camera frame)
            success, rvec, tvec = cv2.solvePnP(marker_points, marker_corners, camera_matrix, dist_coeffs)

            if success:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Compute camera pose relative to marker
                R_inv = R.T
                t_inv = -R_inv @ tvec

                # Convert position to centimeters
                t_inv_cm = t_inv * 100  # meters to cm

                # Convert orientation to Euler angles in degrees
                euler_angles = rotation_matrix_to_euler(R_inv)

                x_cam = t_inv_cm[0][0]
                y_cam = t_inv_cm[1][0]
                yaw_cam = euler_angles[2]

                # Send x_cam, y_cam, yaw_cam over serial
                data_str = f"{x_cam:.1f},{y_cam:.1f},{yaw_cam:.1f}\n"
                try:
                    ser.write(data_str.encode('utf-8'))
                    print(f"Sent over serial: {data_str.strip()}")
                except serial.SerialException as e:
                    print(f"Error sending data: {e}")

                # Format the pose text (fixed typo from original)
                text = f"X[{x_cam:.1f}] cm, Y[{y_cam:.1f}] cm, Yaw[{yaw_cam:.1f}] deg"
                pose_texts.append(text)

                # Draw coordinate axes on the marker (optional visualization)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # Draw detected markers with IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Overlay pose information on the video feed
    for j, text in enumerate(pose_texts):
        cv2.putText(frame, text, (10, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame (comment out if no display is available)
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press (if display is available)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
ser.close()
print("Camera stopped and serial port closed.")