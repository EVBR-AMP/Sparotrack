# Global Shutter Camera Tester v1.1
# ----------------------------------------------
# Quickly try out exposure, gain and other controls on
# Raspberry Pi’s Global Shutter Camera (Sony IMX296) using Picamera2.
#
# • Live preview (OpenCV window)
# • Track-bars for ExposureTime (µs), AnalogueGain (×) and Auto-Exposure toggle
# • Hot-keys:
#     q – quit
#     s – save current frame as PNG in the current folder
# ----------------------------------------------

import os
import sys
import time
from pathlib import Path

import cv2
from picamera2 import Picamera2, Preview


# ---------- Sanity-check: is an OpenCV GUI backend available? ----------
# This avoids mysterious "NULL window handler" errors when OpenCV was
# installed *headless* or you are running without a display.

def _require_highgui():
    try:
        cv2.namedWindow("_test_")
        cv2.destroyWindow("_test_")
    except cv2.error as e:
        print("\nERROR — OpenCV GUI functions are not available.\n"
              "• If you installed 'opencv-python-headless', replace it with 'opencv-python'.\n"
              "  sudo pip uninstall opencv-python-headless && sudo pip install opencv-python\n"
              "• Or install the Debian build that includes GTK / Qt:\n"
              "  sudo apt install python3-opencv\n"
              "• Make sure you have a DISPLAY (run locally, via VNC, or use ssh -X).\n")
        sys.exit(1)

_require_highgui()

# ---------- Initial camera setup ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
)
picam2.configure(config)
picam2.start()

# Query initial defaults from metadata once the camera is running
meta = picam2.capture_metadata()
initial_exposure = int(meta.get("ExposureTime", 3000))  # µs
initial_gain = meta.get("AnalogueGain", 1.0)
auto_exp_enabled = True

# ---------- OpenCV UI setup ----------
window = "Global Shutter Preview"
cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()  # improves responsiveness on some X11/Wayland builds


def nothing(_):
    pass

MAX_EXPOSURE_US = 15000  # ≈1/67 s @ 60 fps
MAX_GAIN_X10 = 160       # 16.0× analogue gain (slider uses ×10)

cv2.createTrackbar("Exposure (µs)", window, initial_exposure, MAX_EXPOSURE_US, nothing)
cv2.createTrackbar("Gain (×10)", window, int(initial_gain * 10), MAX_GAIN_X10, nothing)
cv2.createTrackbar("Auto Exp", window, 1, 1, nothing)


# ---------- Main loop ----------
print("Press q to quit, s to save a PNG snapshot…")
output_dir = Path.cwd()

while True:
    frame = picam2.capture_array("main")

    # ----- UI: read sliders -----
    auto_exp_enabled_new = cv2.getTrackbarPos("Auto Exp", window) == 1
    exposure_slider = cv2.getTrackbarPos("Exposure (µs)", window)
    gain_slider = cv2.getTrackbarPos("Gain (×10)", window)

    if auto_exp_enabled_new != auto_exp_enabled:
        auto_exp_enabled = auto_exp_enabled_new
        picam2.set_controls({"AeEnable": auto_exp_enabled})

    if not auto_exp_enabled:
        picam2.set_controls({
            "ExposureTime": int(exposure_slider),
            "AnalogueGain": gain_slider / 10.0,
        })

    cv2.imshow(window, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        fname = output_dir / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(fname), frame)
        print(f"Saved {fname}")

# ---------- Clean-up ----------
cv2.destroyAllWindows()
picam2.stop()
print("Bye!")
