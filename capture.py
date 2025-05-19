from picamera2 import Picamera2
import cv2, os

save_dir = 'calibration_images'
os.makedirs(save_dir, exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (1920, 1080)}))
picam2.start()

print("SPACE = capture   ESC = quit")
counter = 0
while True:
    frame = picam2.capture_array()
    cv2.imshow("Calibration Capture", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 32:
        path = os.path.join(save_dir, f"calib_{counter:02d}.jpg")
        cv2.imwrite(path, frame)
        print("Saved", path)
        counter += 1

cv2.destroyAllWindows()
picam2.close()
