"""
PTZ camera calibration via ONVIF protocol.

Connects to a PTZ camera, moves it to known angular offsets, captures frames
at each position, and computes a pixel-to-angle calibration matrix.  The
resulting calibration JSON is used by the tracking and preset scripts.

Steps performed:
  1. Connect to camera via ONVIF (or print fallback instructions).
  2. Move camera to a grid of known pan/tilt offsets.
  3. Capture a frame at each position, detect a reference target.
  4. Compute pan_per_pixel and tilt_per_pixel from the displacement.
  5. Save calibration JSON matching the format in DATA_README.txt.

Usage:
    python calibrate_camera.py --camera-ip 192.168.1.101 --mode table \
        --output ../data/calibration/or1_cam1_table.json

    python calibrate_camera.py --camera-ip 192.168.1.101 --mode room \
        --output ../data/calibration/or1_cam2_room.json
"""

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared library import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Optional ONVIF import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from onvif import ONVIFCamera  # onvif-zeep or python-onvif-zeep
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np  # required for calibration math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "config" / "config.yaml"
)

# Calibration grid: small angular steps around the home position.
CALIBRATION_STEPS_PAN = [-2.0, -1.0, 0.0, 1.0, 2.0]   # degrees
CALIBRATION_STEPS_TILT = [-2.0, -1.0, 0.0, 1.0, 2.0]   # degrees

SETTLE_TIME_S = 1.5  # seconds to wait after a PTZ move


# ---------------------------------------------------------------------------
# ONVIF helpers
# ---------------------------------------------------------------------------

def connect_camera(ip: str, port: int = 80, user: str = "admin",
                   password: str = "admin"):
    """Connect to an ONVIF camera and return (camera, ptz_service, media_service, profile)."""
    if not ONVIF_AVAILABLE:
        return None, None, None, None

    cam = ONVIFCamera(ip, port, user, password)
    media = cam.create_media_service()
    ptz = cam.create_ptz_service()
    profiles = media.GetProfiles()
    profile = profiles[0]
    return cam, ptz, media, profile


def move_absolute(ptz, profile, pan_deg: float, tilt_deg: float,
                  zoom: float = 1.0, speed: float = 0.5):
    """Send an absolute PTZ move and wait for it to settle."""
    request = ptz.create_type("AbsoluteMove")
    request.ProfileToken = profile.token
    request.Position = {
        "PanTilt": {"x": pan_deg / 180.0, "y": tilt_deg / 90.0},
        "Zoom": {"x": zoom},
    }
    request.Speed = {
        "PanTilt": {"x": speed, "y": speed},
        "Zoom": {"x": speed},
    }
    ptz.AbsoluteMove(request)
    time.sleep(SETTLE_TIME_S)


def capture_frame_onvif(media, profile):
    """Grab a single frame via the ONVIF snapshot URI."""
    snapshot = media.GetSnapshotUri({"ProfileToken": profile.token})
    uri = snapshot.Uri
    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(uri)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    return None


def capture_frame_rtsp(ip: str):
    """Fallback: grab a frame via RTSP if available."""
    if not CV2_AVAILABLE:
        return None
    uri = f"rtsp://{ip}/stream1"
    cap = cv2.VideoCapture(uri)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------

def detect_reference_centroid(frame):
    """
    Detect the centroid of a calibration target (checkerboard or bright marker)
    in the captured frame.  Falls back to simple bright-spot detection.

    Returns (cx, cy) in pixels or None.
    """
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try checkerboard first
    found, corners = cv2.findChessboardCorners(gray, (7, 5), None)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cx = float(np.mean(corners[:, 0, 0]))
        cy = float(np.mean(corners[:, 0, 1]))
        return cx, cy

    # Fallback: find brightest blob
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            return cx, cy

    return None


def compute_calibration(measurements: list, resolution: tuple):
    """
    Given a list of {pan_deg, tilt_deg, cx, cy} measurements, fit a linear
    model to compute pan_per_pixel and tilt_per_pixel.

    Returns (pan_per_pixel, tilt_per_pixel).
    """
    if len(measurements) < 3:
        # Too few points — use safe defaults for 1080p
        print("Warning: insufficient calibration points, using defaults.")
        return 0.05, 0.04

    pans = np.array([m["pan_deg"] for m in measurements])
    tilts = np.array([m["tilt_deg"] for m in measurements])
    cxs = np.array([m["cx"] for m in measurements])
    cys = np.array([m["cy"] for m in measurements])

    # Linear regression: pan = a * cx + b
    if np.std(cxs) > 0:
        pan_per_pixel = float(np.abs(np.polyfit(cxs, pans, 1)[0]))
    else:
        pan_per_pixel = 0.05

    if np.std(cys) > 0:
        tilt_per_pixel = float(np.abs(np.polyfit(cys, tilts, 1)[0]))
    else:
        tilt_per_pixel = 0.04

    return pan_per_pixel, tilt_per_pixel


def simulate_calibration(mode: str, resolution: tuple = (1920, 1080)):
    """
    Simulate a calibration run when the camera is not available.
    Generates plausible calibration values for testing.
    """
    print("Running simulated calibration (no camera connected)...")
    w, h = resolution

    # Typical values for a 1080p PTZ camera
    fov_h = 60.0 if mode == "room" else 30.0  # degrees horizontal
    fov_v = 34.0 if mode == "room" else 17.0   # degrees vertical

    pan_per_pixel = fov_h / w
    tilt_per_pixel = fov_v / h

    print(f"  Simulated FOV: {fov_h:.1f} x {fov_v:.1f} degrees")
    print(f"  pan_per_pixel:  {pan_per_pixel:.6f}")
    print(f"  tilt_per_pixel: {tilt_per_pixel:.6f}")

    return pan_per_pixel, tilt_per_pixel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PTZ camera calibration via ONVIF protocol."
    )
    parser.add_argument(
        "--camera-ip", type=str, default=None,
        help="IP address of the ONVIF camera (omit to run simulated)"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["table", "room"],
        help="Calibration mode: 'table' (narrow tracking) or 'room' (wide overview)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save calibration JSON"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--port", type=int, default=80,
        help="ONVIF port (default: 80)"
    )
    parser.add_argument(
        "--user", type=str, default="admin",
        help="Camera username (default: admin)"
    )
    parser.add_argument(
        "--password", type=str, default="admin",
        help="Camera password (default: admin)"
    )
    args = parser.parse_args()

    # Load project config for default values
    config = load_config(args.config)
    camera_cfg = config.get("camera", {})
    calib_cfg = config.get("calibration", {})
    tracking_cfg = config.get("table_mode", {}).get("tracking", {})
    dead_zone = tracking_cfg.get("dead_zone_pixels", 30)

    resolution = (1920, 1080)  # default; updated from camera if available

    # ------------------------------------------------------------------
    # Case 1: No camera IP — run simulated calibration
    # ------------------------------------------------------------------
    if args.camera_ip is None:
        pan_pp, tilt_pp = simulate_calibration(args.mode, resolution)
        pid_params = {"kp": 0.5, "ki": 0.01, "kd": 0.1}

    # ------------------------------------------------------------------
    # Case 2: ONVIF not installed — print instructions
    # ------------------------------------------------------------------
    elif not ONVIF_AVAILABLE:
        print("=" * 60)
        print("ONVIF library not available.")
        print()
        print("Install with:")
        print("  pip install onvif-zeep")
        print()
        print("Or run without --camera-ip to generate a simulated calibration.")
        print("=" * 60)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Case 3: Live calibration
    # ------------------------------------------------------------------
    else:
        if not CV2_AVAILABLE:
            print("Error: OpenCV (cv2) is required for live calibration.")
            print("  pip install opencv-python")
            sys.exit(1)

        print(f"Connecting to camera at {args.camera_ip}:{args.port} ...")
        cam, ptz, media, profile = connect_camera(
            args.camera_ip, args.port, args.user, args.password
        )
        if ptz is None:
            print("Failed to connect. Check IP, credentials, and network.")
            sys.exit(1)

        print("Connected. Starting calibration grid...")
        measurements = []

        for pan_deg in CALIBRATION_STEPS_PAN:
            for tilt_deg in CALIBRATION_STEPS_TILT:
                print(f"  Moving to pan={pan_deg:+.1f}, tilt={tilt_deg:+.1f} ...")
                move_absolute(ptz, profile, pan_deg, tilt_deg)

                frame = capture_frame_onvif(media, profile)
                if frame is None:
                    frame = capture_frame_rtsp(args.camera_ip)
                if frame is None:
                    print("    Warning: could not capture frame, skipping.")
                    continue

                resolution = (frame.shape[1], frame.shape[0])
                centroid = detect_reference_centroid(frame)
                if centroid is None:
                    print("    Warning: no target detected, skipping.")
                    continue

                cx, cy = centroid
                measurements.append({
                    "pan_deg": pan_deg, "tilt_deg": tilt_deg,
                    "cx": cx, "cy": cy,
                })
                print(f"    Centroid at ({cx:.1f}, {cy:.1f})")

        print(f"\nCollected {len(measurements)} valid measurements.")
        pan_pp, tilt_pp = compute_calibration(measurements, resolution)

        # Derive PID params from config defaults
        pid_params = {"kp": 0.5, "ki": 0.01, "kd": 0.1}

        # Return camera to home
        print("Returning camera to home position...")
        move_absolute(ptz, profile, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Build and save calibration JSON
    # ------------------------------------------------------------------
    camera_id = "simulated" if args.camera_ip is None else args.camera_ip.replace(".", "_")

    calibration = {
        "camera_id": camera_id,
        "mode": args.mode,
        "ip_address": args.camera_ip or "simulated",
        "protocol": "ONVIF",
        "resolution": list(resolution),
        "pixel_to_angle": {
            "pan_per_pixel": round(pan_pp, 6),
            "tilt_per_pixel": round(tilt_pp, 6),
        },
        "dead_zone_pixels": dead_zone,
        "ptz_limits": {
            "pan_min": -170, "pan_max": 170,
            "tilt_min": -30, "tilt_max": 30,
            "zoom_min": 1.0, "zoom_max": 10.0,
        },
        "pid_params": pid_params,
        "calibrated_by": calib_cfg.get("performed_by", "unknown"),
        "calibrated_date": str(date.today()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to {output_path}")
    print(f"  pan_per_pixel:  {calibration['pixel_to_angle']['pan_per_pixel']}")
    print(f"  tilt_per_pixel: {calibration['pixel_to_angle']['tilt_per_pixel']}")
    print(f"  dead_zone:      {calibration['dead_zone_pixels']} px")


if __name__ == "__main__":
    main()
