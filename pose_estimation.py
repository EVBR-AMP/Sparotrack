import cv2
import numpy as np
from picamera2 import Picamera2

# Camera parameters (placeholders for a 640x480 resolution)
# Replace these with actual calibration values for your Raspberry Pi camera
fx, fy = 640.0, 640.0  # Focal lengths in pixels
cx, cy = 320.0, 240.0  # Principal point (image center)
k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0  # Distortion coefficients (assuming no distortion)

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Define the marker size in meters (e.g., a 10 cm square marker)
marker_length = 0.1

# 3D points of the marker in its own coordinate system (Z=0 plane)
marker_points = np.array([
    [-marker_length / 2, -marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [-marker_length / 2,  marker_length / 2, 0]
], dtype=np.float32)

# Set up ArUco dictionary (e.g., 4x4 markers with 50 IDs)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Create ArUco detector
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
picam2.start()

print("Camera started. Press 'q' to stop (if display is available) or Ctrl+C in terminal.")

# Main loop
try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()
        if frame is None:
            print("Error: Could not read frame.")
            break

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                # Get the 2D corners of the current marker
                marker_corners = corners[i][0]

                # Estimate pose using solvePnP (marker pose in camera frame)
                success, rvec, tvec = cv2.solvePnP(marker_points, marker_corners, camera_matrix, dist_coeffs)

                if success:
                    # Compute the camera's pose relative to the marker
                    R, _ = cv2.Rodrigues(rvec)  # Rotation matrix from rotation vector
                    R_inv = R.T  # Inverse rotation (camera orientation in marker frame)
                    t_inv = -R_inv @ tvec  # Camera position in marker frame
                    rvec_inv, _ = cv2.Rodrigues(R_inv)  # Convert back to rotation vector

                    # Print camera's pose information
                    print(f"Marker ID {ids[i][0]}:")
                    print(f"  Camera Position (m): {t_inv.flatten()}")
                    print(f"  Camera Rotation Vector (rad): {rvec_inv.flatten()}")

                    # Draw coordinate axes on the marker (shows marker's pose in camera frame)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Draw detected markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            print("No markers detected")

        # Display the frame
        cv2.imshow("Frame", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

# Cleanup
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")