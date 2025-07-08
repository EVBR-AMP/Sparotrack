from picamera2 import Picamera2
import cv2, os

save_dir = 'calibration_images'
os.makedirs(save_dir, exist_ok=True)

picam2 = Picamera2()

# --- 1.  Ask explicitly for the 2×2-binned sensor mode ---------------
# (Bookworm & later)                      (Bullseye fallback)
cfg = picam2.create_preview_configuration(
        sensor={'output_size': (2304, 1296)},          # ← full FOV, 2×2 bin
        # or raw={'format': 'SBGGR10_CSI2P', 'size': (2304, 1296)},  # Bullseye
        main={'format': 'BGR888', 'size': (1152, 648)})  # keep 16:9 aspect

picam2.configure(cfg)
# ---------------------------------------------------------------------

picam2.start()

picam2.set_controls({          # your manual-exposure block
    "AeEnable": False,
    "ExposureTime": 4000,      # µs
    "AnalogueGain": 6.7,
})

print("SPACE = capture   ESC = quit")
counter = 0
while True:
    frame = picam2.capture_array()
    cv2.imshow("Calibration Capture", frame)
    k = cv2.waitKey(1)
    if k == 27:           # ESC
        break
    elif k == 32:         # SPACE
        path = os.path.join(save_dir, f"calib_{counter:02d}.jpg")
        cv2.imwrite(path, frame)
        print("Saved", path)
        counter += 1

cv2.destroyAllWindows()
picam2.close()
