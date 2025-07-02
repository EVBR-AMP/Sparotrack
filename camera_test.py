# Global Shutter Camera Tester
# ----------------------------------------------
# Quickly try out exposure, gain and other controls on
# Raspberry Pi’s Global Shutter Camera (Sony IMX296) using Picamera2.
#
# • Live preview (OpenCV window)
# • Track‑bars for ExposureTime (µs), AnalogueGain (×) and Auto‑Exposure toggle
# • Hot‑keys:
#     q – quit
#     s – save current frame as PNG in the current folder
# ----------------------------------------------

import time
from pathlib import Path

import cv2
from picamera2 import Picamera2, Preview


# ---------- Initial camera setup ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
)
picam2.configure(config)

# Query initial defaults from metadata once the camera is running
picam2.start()
meta = picam2.capture_metadata()  # first frame’s metadata

# Fallback values in case keys are missing (shouldn’t happen on Pi 5 kernel ≥ 6.9)
initial_exposure = int(meta.get("ExposureTime", 3000))  # µs
initial_gain = meta.get("AnalogueGain", 1.0)

auto_exp_enabled = True


# ---------- OpenCV UI setup ----------
window = "Global Shutter Preview"
cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)


def nothing(_):
    pass


# Track‑bar ranges – adjust to taste or sensor limits
MAX_EXPOSURE_US = 15000  # ≈ 1/67 s @ 60 fps, change if you use lower FrameDurationLimits
MAX_GAIN_X10 = 160       # 16.0× analogue gain, expressed ×10 to keep an int slider

cv2.createTrackbar("Exposure (µs)", window, initial_exposure, MAX_EXPOSURE_US, nothing)
cv2.createTrackbar("Gain (×10)", window, int(initial_gain * 10), MAX_GAIN_X10, nothing)
cv2.createTrackbar("Auto Exp", window, 1 if auto_exp_enabled else 0, 1, nothing)


# ---------- Main loop ----------
print("Press q to quit, s to save a PNG snapshot…")
frame_counter = 0
output_dir = Path.cwd()

while True:
    # Fetch frame *before* handling UI so we keep the preview smooth
    frame = picam2.capture_array("main")  # numpy RGB888

    # ----- UI: read sliders -----
    auto_exp_enabled_new = cv2.getTrackbarPos("Auto Exp", window) == 1
    exposure_slider = cv2.getTrackbarPos("Exposure (µs)", window)
    gain_slider = cv2.getTrackbarPos("Gain (×10)", window)

    # Apply any changes (avoid spamming set_controls every loop)
    if auto_exp_enabled_new != auto_exp_enabled:
        auto_exp_enabled = auto_exp_enabled_new
        picam2.set_controls({"AeEnable": auto_exp_enabled})

    if not auto_exp_enabled:
        # Note: ExposureTime expects integer µs; AnalogueGain is a float
        picam2.set_controls({
            "ExposureTime": int(exposure_slider),
            "AnalogueGain": gain_slider / 10.0,
        })

    # ----- Display frame -----
    cv2.imshow(window, frame)

    # Handle key‑presses (non‑blocking)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fname = output_dir / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(fname), frame)
        print(f"Saved {fname}")

    frame_counter += 1

# ---------- Clean‑up ----------
cv2.destroyAllWindows()
picam2.stop()
print("Bye!")
