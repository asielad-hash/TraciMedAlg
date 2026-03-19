================================================================
04 — CAMERA CONTROLLER
PTZ Auto-Aim: Table Mode + Room Mode
================================================================

WHAT THIS COMPONENT DOES:
  Controls PTZ cameras in two modes:

  MODE A — TABLE MODE:
    Camera auto-aims to keep the mayo/tray table centered.
    Uses Kalman filter for smooth tracking + PID controller
    to prevent oscillation. Feeds video to Instrument Tracker (#01)
    and Tray Monitor (#02).

  MODE B — ROOM MODE:
    Wide-angle overview of the entire OR. Feeds video to
    Surgical Workflow (#03) for phase and idle detection.
    Switches between preset positions based on surgical phase.
    Can zoom in to provide table feeds when dedicated cameras
    are not available.

  TYPE: Classic algorithms (no ML training needed)
  REQUIRES: Per-camera calibration at installation

================================================================
THIS COMPONENT HAS NO TRAINING DATA
================================================================

  Camera control uses classic algorithms:
    - Kalman filter (smooth tracking)
    - PID controller (prevent oscillation)
    - ONVIF PTZ protocol
    - Phase-driven preset switching (rules from config)

  The only "data" needed is per-camera calibration:

================================================================
CALIBRATION DATA
================================================================

data/
├── calibration/                ← Per-camera calibration files
│   ├── or1_cam1_table.json     ← table-mode camera in OR #1
│   ├── or1_cam2_room.json      ← room-mode camera in OR #1
│   └── ...
│
└── roi_configs/                ← Region of interest definitions
    ├── or1_mayo_table_roi.json
    ├── or1_back_table_roi.json
    └── ...

----------------------------------------------------------------
CALIBRATION FILE FORMAT
----------------------------------------------------------------

  Per camera: {or}_{camera}_{mode}.json

  {
    "camera_id": "or1_cam1",
    "mode": "table",
    "ip_address": "192.168.1.101",
    "protocol": "ONVIF",
    "resolution": [1920, 1080],
    "pixel_to_angle": {
      "pan_per_pixel": 0.05,
      "tilt_per_pixel": 0.04
    },
    "dead_zone_pixels": 30,
    "ptz_limits": {
      "pan_min": -170, "pan_max": 170,
      "tilt_min": -30, "tilt_max": 30,
      "zoom_min": 1.0, "zoom_max": 10.0
    },
    "pid_params": {
      "kp": 0.5, "ki": 0.01, "kd": 0.1
    },
    "calibrated_by": "technician_name",
    "calibrated_date": "2026-03-15"
  }

----------------------------------------------------------------
ROI CONFIG FORMAT
----------------------------------------------------------------

  Per table: {or}_{table_type}_roi.json

  {
    "or_id": "or1",
    "table_type": "mayo_table",
    "roi_pixels": [100, 50, 800, 600],
    "roi_notes": "Mayo table, surgeon's right side"
  }

================================================================
HOW TO SET UP
================================================================

  1. Install camera in OR
  2. Connect via ONVIF
  3. Run calibration script:

     python scripts/calibrate_camera.py \
       --camera-ip 192.168.1.101 \
       --mode table \
       --output data/calibration/or1_cam1_table.json

  4. Define table ROI:

     python scripts/define_roi.py \
       --camera-ip 192.168.1.101 \
       --output data/roi_configs/or1_mayo_table_roi.json

  5. Test tracking:

     python scripts/test_tracking.py \
       --calibration data/calibration/or1_cam1_table.json

================================================================
TESTING CHECKLIST
================================================================

  [ ] Camera connects via ONVIF
  [ ] Pan/tilt/zoom commands execute correctly
  [ ] Kalman filter tracks table centroid smoothly
  [ ] PID prevents oscillation on minor jitter
  [ ] Dead zone prevents unnecessary movement
  [ ] Room mode presets reach correct positions
  [ ] Phase-driven switching works with workflow input
  [ ] Latency from detection to PTZ move < 200ms
================================================================
