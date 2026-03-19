"""
Interactive ROI drawing tool for defining table regions of interest.

Opens a camera frame (live via RTSP or loaded from an image file) in an
OpenCV window.  The user draws a rectangle with the mouse to define the
table ROI.  The ROI is saved as JSON in the format expected by the camera
controller and described in DATA_README.txt.

Usage:
    python define_roi.py --camera-ip 192.168.1.101 \
        --or-id or1 --table-type mayo_table \
        --output ../data/roi_configs/or1_mayo_table_roi.json

    python define_roi.py --image-path frame.jpg \
        --or-id or1 --table-type back_table \
        --output ../data/roi_configs/or1_back_table_roi.json
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared library import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config  # noqa: E402

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_NAME = "Define ROI - Draw rectangle, press ENTER to confirm, ESC to cancel"

TABLE_TYPES = ["mayo_table", "back_table", "tray_table", "instrument_table"]


# ---------------------------------------------------------------------------
# Mouse callback for rectangle drawing
# ---------------------------------------------------------------------------

class RoiDrawer:
    """Manages interactive rectangle drawing on an OpenCV window."""

    def __init__(self, frame):
        self.original = frame.copy()
        self.frame = frame.copy()
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.roi_set = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for rectangle drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y
            self.end_x = x
            self.end_y = y
            self.roi_set = False

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_x = x
            self.end_y = y
            # Redraw with current rectangle
            self.frame = self.original.copy()
            cv2.rectangle(
                self.frame,
                (self.start_x, self.start_y),
                (self.end_x, self.end_y),
                (0, 255, 0), 2,
            )

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x = x
            self.end_y = y
            self.roi_set = True
            # Final rectangle
            self.frame = self.original.copy()
            cv2.rectangle(
                self.frame,
                (self.start_x, self.start_y),
                (self.end_x, self.end_y),
                (0, 255, 0), 2,
            )

    def get_roi(self):
        """Return ROI as [x, y, w, h] with top-left origin, or None."""
        if not self.roi_set:
            return None
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)
        w = x2 - x1
        h = y2 - y1
        if w < 5 or h < 5:
            return None
        return [x1, y1, w, h]


# ---------------------------------------------------------------------------
# Frame acquisition
# ---------------------------------------------------------------------------

def grab_frame_from_camera(ip: str):
    """Capture a single frame from a camera via RTSP."""
    if not CV2_AVAILABLE:
        return None
    uri = f"rtsp://{ip}/stream1"
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        # Try alternative URI patterns
        for alt in [f"rtsp://{ip}:554/stream1", f"rtsp://admin:admin@{ip}/stream1"]:
            cap = cv2.VideoCapture(alt)
            if cap.isOpened():
                break
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def load_frame_from_file(path: str):
    """Load a frame from an image file."""
    if not CV2_AVAILABLE:
        return None
    frame = cv2.imread(path)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI drawing tool for table region definition."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--camera-ip", type=str, default=None,
        help="IP address of camera to grab a live frame from"
    )
    source.add_argument(
        "--image-path", type=str, default=None,
        help="Path to an image file to use instead of live camera"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save ROI JSON"
    )
    parser.add_argument(
        "--or-id", type=str, required=True,
        help="Operating room identifier (e.g., 'or1')"
    )
    parser.add_argument(
        "--table-type", type=str, required=True,
        choices=TABLE_TYPES,
        help="Type of table being defined"
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Optional notes about this ROI (e.g., 'Mayo table, surgeon right side')"
    )
    args = parser.parse_args()

    if not CV2_AVAILABLE:
        print("Error: OpenCV (cv2) is required for interactive ROI drawing.")
        print("  pip install opencv-python")
        sys.exit(1)

    # Acquire frame
    if args.camera_ip:
        print(f"Grabbing frame from camera at {args.camera_ip} ...")
        frame = grab_frame_from_camera(args.camera_ip)
        if frame is None:
            print("Error: could not grab frame from camera.")
            print("Check the IP address, network connection, and RTSP availability.")
            sys.exit(1)
    else:
        print(f"Loading image from {args.image_path} ...")
        frame = load_frame_from_file(args.image_path)
        if frame is None:
            print(f"Error: could not load image from {args.image_path}")
            sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame size: {w} x {h}")
    print()
    print("Instructions:")
    print("  - Click and drag to draw the table ROI rectangle.")
    print("  - Press ENTER to confirm the selection.")
    print("  - Press 'r' to reset and redraw.")
    print("  - Press ESC to cancel without saving.")
    print()

    # Interactive drawing loop
    drawer = RoiDrawer(frame)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(WINDOW_NAME, drawer.mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, drawer.frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("Cancelled. No ROI saved.")
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord("r"):
            drawer = RoiDrawer(frame)
            cv2.setMouseCallback(WINDOW_NAME, drawer.mouse_callback)
            print("Reset. Draw a new rectangle.")

        elif key == 13:  # ENTER
            roi = drawer.get_roi()
            if roi is None:
                print("No valid ROI drawn. Draw a rectangle first.")
                continue
            break

    cv2.destroyAllWindows()

    # Build ROI JSON
    roi_data = {
        "or_id": args.or_id,
        "table_type": args.table_type,
        "roi_pixels": roi,
        "roi_notes": args.notes or f"{args.table_type} in {args.or_id}",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(roi_data, f, indent=2)

    x, y, rw, rh = roi
    print(f"\nROI saved to {output_path}")
    print(f"  or_id:      {args.or_id}")
    print(f"  table_type: {args.table_type}")
    print(f"  roi_pixels: x={x}, y={y}, w={rw}, h={rh}")


if __name__ == "__main__":
    main()
