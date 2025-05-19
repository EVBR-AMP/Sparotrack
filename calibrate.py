"""
calibrate.py
Compute camera intrinsics from a set of chess-board photos.

∙ Works with OpenCV ≥ 4
∙ Produces camera.yaml you can load in any later script

Usage example
-------------
python3 calibrate.py \
        --dir calibration_images \
        --rows 6 --cols 9 --square-size 25 \
        --output camera.yaml --preview
"""

import argparse, glob, os, datetime, yaml
import cv2
import numpy as np

# ---------- command-line -----------------------------------------------------

P = argparse.ArgumentParser(description="Calibrate camera from chess-board images")
P.add_argument("--dir", default="calibration_images",
               help="folder containing the snapshots")
P.add_argument("--rows", type=int, default=6, help="inner corners vertically")
P.add_argument("--cols", type=int, default=9, help="inner corners horizontally")
P.add_argument("--square-size", type=float, default=25.0,
               help="square size in mm (or any consistent unit)")
P.add_argument("--output", default="camera.yaml", help="YAML file to write")
P.add_argument("--preview", action="store_true",
               help="show detected corners while processing")

args = P.parse_args()

# ---------- prepare object-point template ------------------------------------

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
objp *= args.square_size

obj_points, img_points = [], []

# ---------- loop over files --------------------------------------------------

image_paths = sorted(
    glob.glob(os.path.join(args.dir, "*.jpg")) +
    glob.glob(os.path.join(args.dir, "*.png"))
)

if not image_paths:
    raise SystemExit(f"No images found in {args.dir}")

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
    if not ok:
        print(f"✗ corners NOT found in {os.path.basename(path)}")
        continue

    cv2.cornerSubPix(
        gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    obj_points.append(objp)
    img_points.append(corners)

    if args.preview:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners, ok)
        cv2.imshow("preview", vis)
        cv2.waitKey(300)

cv2.destroyAllWindows()

if len(obj_points) < 5:
    raise SystemExit("Need at least five good images – capture more!")

# ---------- run calibration --------------------------------------------------

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# mean reprojection error
mean_err = np.mean([
    cv2.norm(img_points[i],
             cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)[0],
             cv2.NORM_L2) / len(img_points[i])
    for i in range(len(obj_points))
])

# ---------- save to YAML -----------------------------------------------------

calib = {
    "image_size": list(gray.shape[::-1]),   # [w, h]
    "camera_matrix": K.tolist(),
    "dist_coeff": dist.ravel().tolist(),
    "reprojection_error": float(mean_err),
    "pattern": {
        "rows": args.rows,
        "cols": args.cols,
        "square_size": args.square_size
    },
    "date": datetime.datetime.now().isoformat(timespec="seconds")
}

with open(args.output, "w") as f:
    yaml.safe_dump(calib, f)

print(f"✅ camera.yaml written  (RMS err ≈ {mean_err:.4f} px)")
