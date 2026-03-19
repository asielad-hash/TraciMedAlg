"""
Kalman filter + PID controller tracking test for PTZ camera control.

Loads a calibration JSON, then runs a Kalman-filter-smoothed centroid
tracker paired with a PID controller.  Can operate on:
  - A test video file (detect a target and track it)
  - Simulated centroid data (synthetic noisy trajectory)

Reports tracking metrics: smoothness, dead-zone effectiveness, and
latency estimates.

Usage:
    python test_tracking.py --calibration ../data/calibration/or1_cam1_table.json

    python test_tracking.py --calibration ../data/calibration/or1_cam1_table.json \
        --video test_video.mp4

    python test_tracking.py --calibration ../data/calibration/or1_cam1_table.json \
        --simulate --num-frames 500
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "config" / "config.yaml"
)


# ---------------------------------------------------------------------------
# Kalman Filter for 2D centroid tracking
# ---------------------------------------------------------------------------

class CentroidKalmanFilter:
    """
    Simple 2D Kalman filter for tracking a centroid (cx, cy).

    State vector: [cx, cy, vx, vy]  (position + velocity)
    Measurement:  [cx, cy]
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 5.0):
        self.dt = 1.0  # normalized time step

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise
        self.Q = np.eye(4) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise

        # State and covariance
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0

        self.initialized = False

    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement):
        """Update state with a new measurement [cx, cy]."""
        z = np.array(measurement, dtype=np.float64)

        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return self.x[:2].copy()

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].copy()


# ---------------------------------------------------------------------------
# PID Controller for PTZ movement
# ---------------------------------------------------------------------------

class PIDController:
    """
    PID controller that converts pixel error to PTZ velocity commands.

    Operates independently on pan (x-axis) and tilt (y-axis).
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1,
                 dead_zone: float = 30.0, output_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dead_zone = dead_zone
        self.output_limit = output_limit

        # Internal state (separate for x and y)
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

    def compute(self, error_x: float, error_y: float, dt: float = 1.0):
        """
        Compute PID output for the given pixel errors.

        Returns (cmd_pan, cmd_tilt).  Values are zero when error is
        within the dead zone.
        """
        # Dead zone — suppress small errors to prevent jitter
        if abs(error_x) < self.dead_zone:
            error_x = 0.0
        if abs(error_y) < self.dead_zone:
            error_y = 0.0

        # PID for X (pan)
        self.integral_x += error_x * dt
        derivative_x = (error_x - self.prev_error_x) / dt if dt > 0 else 0.0
        cmd_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
        self.prev_error_x = error_x

        # PID for Y (tilt)
        self.integral_y += error_y * dt
        derivative_y = (error_y - self.prev_error_y) / dt if dt > 0 else 0.0
        cmd_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y
        self.prev_error_y = error_y

        # Clamp output
        cmd_x = max(-self.output_limit, min(self.output_limit, cmd_x))
        cmd_y = max(-self.output_limit, min(self.output_limit, cmd_y))

        return cmd_x, cmd_y

    def reset(self):
        """Reset internal state."""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0


# ---------------------------------------------------------------------------
# Simulation: generate noisy centroid trajectory
# ---------------------------------------------------------------------------

def generate_simulated_trajectory(num_frames: int, resolution: tuple,
                                  noise_std: float = 15.0):
    """
    Generate a synthetic centroid trajectory with smooth movement + noise.

    The trajectory follows a slow figure-eight pattern with Gaussian noise
    to simulate a tracked object (table edge) drifting in the frame.
    """
    w, h = resolution
    cx_center = w / 2.0
    cy_center = h / 2.0

    t = np.linspace(0, 4 * np.pi, num_frames)
    # Figure-eight (Lissajous)
    cx_clean = cx_center + (w * 0.2) * np.sin(t)
    cy_clean = cy_center + (h * 0.15) * np.sin(2 * t)

    # Add noise
    cx_noisy = cx_clean + np.random.normal(0, noise_std, num_frames)
    cy_noisy = cy_clean + np.random.normal(0, noise_std, num_frames)

    return list(zip(cx_noisy.tolist(), cy_noisy.tolist()))


# ---------------------------------------------------------------------------
# Video-based centroid detection
# ---------------------------------------------------------------------------

def detect_centroid_in_frame(frame):
    """Detect the centroid of the largest moving/bright object in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def extract_centroids_from_video(video_path: str):
    """Read video and detect centroids for each frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return [], (1920, 1080)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    centroids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c = detect_centroid_in_frame(frame)
        centroids.append(c)

    cap.release()
    return centroids, (w, h)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(raw_centroids, filtered_positions, pid_commands,
                    dead_zone: float, resolution: tuple):
    """
    Compute and return tracking quality metrics.

    Metrics:
      - smoothness: average frame-to-frame jitter in filtered positions
      - dead_zone_suppression: fraction of frames where PID output was zero
      - max_command: peak PID output (indicates worst-case aggressiveness)
      - rms_error: RMS error between raw and filtered centroids
      - estimated_latency_ms: time from centroid to PID output (always 1 frame)
    """
    w, h = resolution

    # Smoothness: average displacement between consecutive filtered positions
    displacements = []
    for i in range(1, len(filtered_positions)):
        dx = filtered_positions[i][0] - filtered_positions[i - 1][0]
        dy = filtered_positions[i][1] - filtered_positions[i - 1][1]
        displacements.append(math.sqrt(dx * dx + dy * dy))
    avg_smoothness = float(np.mean(displacements)) if displacements else 0.0

    # Dead-zone suppression rate
    zero_count = sum(
        1 for (cx, cy) in pid_commands if abs(cx) < 1e-6 and abs(cy) < 1e-6
    )
    suppression_rate = zero_count / len(pid_commands) if pid_commands else 0.0

    # Max command magnitude
    magnitudes = [math.sqrt(cx ** 2 + cy ** 2) for cx, cy in pid_commands]
    max_cmd = max(magnitudes) if magnitudes else 0.0

    # RMS error between raw and filtered
    errors = []
    for raw, filt in zip(raw_centroids, filtered_positions):
        if raw is not None:
            dx = raw[0] - filt[0]
            dy = raw[1] - filt[1]
            errors.append(dx * dx + dy * dy)
    rms_error = math.sqrt(np.mean(errors)) if errors else 0.0

    return {
        "smoothness_avg_px": round(avg_smoothness, 2),
        "dead_zone_suppression_pct": round(suppression_rate * 100, 1),
        "max_pid_command": round(max_cmd, 4),
        "rms_filter_error_px": round(rms_error, 2),
        "estimated_latency_frames": 1,
        "total_frames": len(raw_centroids),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kalman filter + PID controller tracking test."
    )
    parser.add_argument(
        "--calibration", type=str, required=True,
        help="Path to calibration JSON (from calibrate_camera.py)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to test video file (optional; uses detection-based tracking)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run on simulated centroid data instead of a video"
    )
    parser.add_argument(
        "--num-frames", type=int, default=300,
        help="Number of simulated frames (default: 300, only with --simulate)"
    )
    parser.add_argument(
        "--noise-std", type=float, default=15.0,
        help="Noise standard deviation for simulation (default: 15.0 px)"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    args = parser.parse_args()

    if not args.simulate and args.video is None:
        # Default to simulation if no video provided
        print("No --video provided; defaulting to --simulate mode.")
        args.simulate = True

    # Load calibration
    calib_path = Path(args.calibration)
    if not calib_path.is_file():
        print(f"Error: calibration file not found: {calib_path}")
        print("Run calibrate_camera.py first to generate it.")
        sys.exit(1)

    with open(calib_path) as f:
        calib = json.load(f)

    resolution = tuple(calib.get("resolution", [1920, 1080]))
    dead_zone = calib.get("dead_zone_pixels", 30)
    pid_params = calib.get("pid_params", {"kp": 0.5, "ki": 0.01, "kd": 0.1})

    print(f"Calibration: {calib_path.name}")
    print(f"  Resolution:     {resolution[0]}x{resolution[1]}")
    print(f"  Dead zone:      {dead_zone} px")
    print(f"  PID params:     kp={pid_params['kp']}, ki={pid_params['ki']}, kd={pid_params['kd']}")
    print(f"  pan_per_pixel:  {calib['pixel_to_angle']['pan_per_pixel']}")
    print(f"  tilt_per_pixel: {calib['pixel_to_angle']['tilt_per_pixel']}")
    print()

    # Acquire centroid data
    if args.simulate:
        print(f"Generating simulated trajectory ({args.num_frames} frames, "
              f"noise_std={args.noise_std}) ...")
        raw_centroids = generate_simulated_trajectory(
            args.num_frames, resolution, args.noise_std
        )
    else:
        if not CV2_AVAILABLE:
            print("Error: OpenCV (cv2) is required for video-based tracking.")
            print("  pip install opencv-python")
            sys.exit(1)
        print(f"Extracting centroids from video: {args.video} ...")
        raw_centroids, resolution = extract_centroids_from_video(args.video)
        if not raw_centroids:
            print("Error: no frames extracted from video.")
            sys.exit(1)

    # Load config for tracking parameters
    config = load_config(args.config)
    tracking_cfg = config.get("table_mode", {}).get("tracking", {})
    smooth_factor = tracking_cfg.get("smooth_factor", 0.7)

    # Initialize filter and controller
    kf = CentroidKalmanFilter(
        process_noise=1.0,
        measurement_noise=args.noise_std if args.simulate else 5.0,
    )
    pid = PIDController(
        kp=pid_params["kp"],
        ki=pid_params["ki"],
        kd=pid_params["kd"],
        dead_zone=dead_zone,
        output_limit=1.0,
    )

    frame_center = (resolution[0] / 2.0, resolution[1] / 2.0)

    # Run tracking loop
    filtered_positions = []
    pid_commands = []

    print("Running tracking loop ...")
    t_start = time.perf_counter()

    for i, centroid in enumerate(raw_centroids):
        if centroid is None:
            # No detection — use prediction only
            pred = kf.predict()
            filtered_positions.append((pred[0], pred[1]))
            pid_commands.append((0.0, 0.0))
            continue

        kf.predict()
        filtered = kf.update(centroid)
        filtered_positions.append((filtered[0], filtered[1]))

        # PID: error = offset from frame center
        error_x = filtered[0] - frame_center[0]
        error_y = filtered[1] - frame_center[1]
        cmd_pan, cmd_tilt = pid.compute(error_x, error_y)
        pid_commands.append((cmd_pan, cmd_tilt))

    t_elapsed = time.perf_counter() - t_start

    # Compute metrics
    metrics = compute_metrics(
        raw_centroids, filtered_positions, pid_commands,
        dead_zone, resolution,
    )

    processing_fps = len(raw_centroids) / t_elapsed if t_elapsed > 0 else 0

    # Report results
    print()
    print("=" * 60)
    print("TRACKING TEST RESULTS")
    print("=" * 60)
    print(f"  Total frames processed:      {metrics['total_frames']}")
    print(f"  Processing speed:            {processing_fps:.0f} fps")
    print(f"  Smoothness (avg jitter):     {metrics['smoothness_avg_px']:.2f} px/frame")
    print(f"  Dead-zone suppression:       {metrics['dead_zone_suppression_pct']:.1f}%")
    print(f"  Max PID command:             {metrics['max_pid_command']:.4f}")
    print(f"  RMS filter error:            {metrics['rms_filter_error_px']:.2f} px")
    print(f"  Estimated latency:           {metrics['estimated_latency_frames']} frame(s)")
    print("=" * 60)

    # Quality assessment
    print()
    if metrics["smoothness_avg_px"] < 20.0:
        print("[PASS] Smoothness is within acceptable range.")
    else:
        print("[WARN] High jitter detected. Consider tuning Kalman process noise.")

    if metrics["dead_zone_suppression_pct"] > 10.0:
        print("[PASS] Dead zone is actively suppressing micro-movements.")
    else:
        print("[INFO] Dead zone rarely triggered. Target may be highly mobile.")

    if metrics["max_pid_command"] < 0.8:
        print("[PASS] PID output stays within safe limits.")
    else:
        print("[WARN] PID output near saturation. Consider reducing kp.")

    print()


if __name__ == "__main__":
    main()
