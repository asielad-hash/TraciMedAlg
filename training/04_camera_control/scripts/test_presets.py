"""
Validate room-mode preset positions for PTZ camera control.

Loads the component config.yaml, reads the preset positions defined for
room mode (overview, mayo_table, back_table, surgical_field), commands
the camera to each preset via ONVIF (or simulates the movement), and
captures a frame at each position for visual review.

Also validates the phase-driven switching rules from config to ensure
every surgical phase maps to a valid preset name.

Usage:
    python test_presets.py --config ../config/config.yaml \
        --camera-ip 192.168.1.102

    python test_presets.py --config ../config/config.yaml --simulate
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared library import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config  # noqa: E402

try:
    from onvif import ONVIFCamera
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent / "config" / "config.yaml"
)
SETTLE_TIME_S = 2.0


# ---------------------------------------------------------------------------
# ONVIF helpers
# ---------------------------------------------------------------------------

def connect_camera(ip: str, port: int = 80, user: str = "admin",
                   password: str = "admin"):
    """Connect to an ONVIF camera and return (ptz_service, media_service, profile)."""
    if not ONVIF_AVAILABLE:
        return None, None, None
    cam = ONVIFCamera(ip, port, user, password)
    media = cam.create_media_service()
    ptz = cam.create_ptz_service()
    profiles = media.GetProfiles()
    profile = profiles[0]
    return ptz, media, profile


def move_to_preset(ptz, profile, pan_deg: float, tilt_deg: float,
                   zoom: float, speed: float = 0.5):
    """Move camera to an absolute PTZ position."""
    request = ptz.create_type("AbsoluteMove")
    request.ProfileToken = profile.token
    request.Position = {
        "PanTilt": {"x": pan_deg / 180.0, "y": tilt_deg / 90.0},
        "Zoom": {"x": zoom / 10.0},  # normalize zoom to 0-1 range
    }
    request.Speed = {
        "PanTilt": {"x": speed, "y": speed},
        "Zoom": {"x": speed},
    }
    ptz.AbsoluteMove(request)
    time.sleep(SETTLE_TIME_S)


def capture_frame(media, profile, ip: str):
    """Capture a frame via ONVIF snapshot or RTSP fallback."""
    if not CV2_AVAILABLE:
        return None

    # Try ONVIF snapshot
    try:
        snapshot = media.GetSnapshotUri({"ProfileToken": profile.token})
        cap = cv2.VideoCapture(snapshot.Uri)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    except Exception:
        pass

    # RTSP fallback
    cap = cv2.VideoCapture(f"rtsp://{ip}/stream1")
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_preset_move(name: str, pan: float, tilt: float, zoom: float):
    """Simulate moving to a preset position (prints what would happen)."""
    print(f"    [SIM] AbsoluteMove -> pan={pan:.1f}, tilt={tilt:.1f}, zoom={zoom:.1f}")
    print(f"    [SIM] Waiting {SETTLE_TIME_S}s for settle...")
    # No actual wait in simulation for speed


def generate_simulated_frame(name: str, resolution: tuple = (1920, 1080)):
    """Generate a placeholder frame with the preset name overlaid."""
    if not CV2_AVAILABLE:
        return None
    w, h = resolution
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark blue-grey background
    frame[:, :] = (40, 40, 60)
    # Draw text
    text = f"PRESET: {name.upper()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 3
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = (h + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), font, scale, (0, 255, 0), thickness)
    return frame


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_switching_rules(config: dict, preset_names: set):
    """
    Validate that every phase-driven switching rule maps to a valid preset.

    Returns list of (phase, preset, is_valid) tuples.
    """
    switching = config.get("room_mode", {}).get("switching", {})
    rules = switching.get("rules", {})

    results = []
    for phase, preset in rules.items():
        valid = preset in preset_names
        results.append((phase, preset, valid))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate room-mode preset positions for PTZ camera."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--camera-ip", type=str, default=None,
        help="IP address of the ONVIF camera"
    )
    source.add_argument(
        "--simulate", action="store_true",
        help="Simulate camera movement (no hardware required)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save captured frames (default: ./preset_frames/)"
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

    # Load config
    config = load_config(args.config)

    # Extract preset positions
    room_cfg = config.get("room_mode", {})
    presets = room_cfg.get("coverage", {}).get("preset_positions", [])

    if not presets:
        print("Error: no preset_positions found in config room_mode.coverage.")
        sys.exit(1)

    preset_names = {p["name"] for p in presets}
    print(f"Found {len(presets)} preset positions: {', '.join(sorted(preset_names))}")
    print()

    # Output directory for captured frames
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "preset_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera speed limit from config
    speed_limit = config.get("camera", {}).get("ptz_speed_limit", 0.5)

    # ------------------------------------------------------------------
    # Connect to camera (or simulate)
    # ------------------------------------------------------------------
    ptz, media, profile = None, None, None

    if not args.simulate:
        if not ONVIF_AVAILABLE:
            print("=" * 60)
            print("ONVIF library not available.")
            print()
            print("Install with:")
            print("  pip install onvif-zeep")
            print()
            print("Or use --simulate to run without hardware.")
            print("=" * 60)
            sys.exit(1)

        print(f"Connecting to camera at {args.camera_ip}:{args.port} ...")
        ptz, media, profile = connect_camera(
            args.camera_ip, args.port, args.user, args.password
        )
        if ptz is None:
            print("Failed to connect. Check IP, credentials, and network.")
            sys.exit(1)
        print("Connected.\n")

    # ------------------------------------------------------------------
    # Test each preset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TESTING PRESET POSITIONS")
    print("=" * 60)

    results = []

    for preset in presets:
        name = preset["name"]
        pan = preset["pan"]
        tilt = preset["tilt"]
        zoom = preset["zoom"]

        print(f"\n  [{name}] pan={pan}, tilt={tilt}, zoom={zoom}")

        if args.simulate:
            simulate_preset_move(name, pan, tilt, zoom)
            frame = generate_simulated_frame(name)
            status = "OK (simulated)"
        else:
            try:
                move_to_preset(ptz, profile, pan, tilt, zoom, speed_limit)
                frame = capture_frame(media, profile, args.camera_ip)
                status = "OK" if frame is not None else "WARN: no frame captured"
            except Exception as e:
                frame = None
                status = f"ERROR: {e}"

        # Save frame if available
        if frame is not None and CV2_AVAILABLE:
            frame_path = output_dir / f"preset_{name}.jpg"
            cv2.imwrite(str(frame_path), frame)
            print(f"    Frame saved: {frame_path}")
        elif not CV2_AVAILABLE:
            print("    (OpenCV not available, frame not saved)")

        results.append({"name": name, "status": status})
        print(f"    Status: {status}")

    # ------------------------------------------------------------------
    # Validate phase-driven switching rules
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VALIDATING SWITCHING RULES")
    print("=" * 60)

    rule_results = validate_switching_rules(config, preset_names)

    all_valid = True
    for phase, preset, valid in rule_results:
        mark = "OK" if valid else "FAIL"
        if not valid:
            all_valid = False
        print(f"  [{mark}] {phase:30s} -> {preset}")

    if not rule_results:
        print("  (no switching rules defined)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    preset_ok = sum(1 for r in results if r["status"].startswith("OK"))
    print(f"  Presets tested:   {preset_ok}/{len(results)} passed")
    print(f"  Switching rules:  {'ALL VALID' if all_valid else 'SOME INVALID'}")
    print(f"  Frames saved to:  {output_dir}")

    if preset_ok == len(results) and all_valid:
        print("\n  [PASS] All preset positions and switching rules validated.")
    else:
        print("\n  [WARN] Some tests did not pass. Review output above.")

    print()


if __name__ == "__main__":
    main()
