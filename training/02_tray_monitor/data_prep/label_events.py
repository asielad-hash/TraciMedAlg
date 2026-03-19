"""
Simple OpenCV tool for annotating tray event timestamps in surgical videos.

Plays a video and lets the annotator mark TAKE_OUT and PUT_BACK events
interactively.  The resulting event annotation JSON follows the format
described in data/DATA_README.txt and is consumed by extract_event_clips.py.

Controls:
    SPACE       Pause / resume playback
    o           Mark TAKE_OUT event at current timestamp
    i           Mark PUT_BACK event at current timestamp
    RIGHT       Seek forward 1 second (5 sec while paused with SHIFT not available,
                so press multiple times)
    LEFT        Seek backward 1 second
    +  / -      Increase / decrease playback speed
    r           Set tray ROI (opens ROI selection window)
    u           Undo last annotation
    q / ESC     Quit and save

Usage:
    python label_events.py --video data/videos/sheba_general_001_tray.mp4 --output data/annotations/sheba_general_001_tray_events.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.video_io import get_video_info


def draw_overlay(frame: np.ndarray, info: dict, events: list,
                 paused: bool, speed: float, tray_roi: list | None) -> np.ndarray:
    """Draw HUD overlay showing current time, event count, and controls."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Semi-transparent banner at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    current_sec = info["current_sec"]
    total_sec = info["total_sec"]
    current_frame = info["current_frame"]
    total_frames = info["total_frames"]

    # Time display
    time_str = (f"Time: {current_sec:.1f}s / {total_sec:.1f}s  "
                f"Frame: {current_frame}/{total_frames}")
    cv2.putText(display, time_str, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status
    status = "PAUSED" if paused else f"PLAYING x{speed:.1f}"
    color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(display, status, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Event count
    n_take = sum(1 for e in events if e["type"] == "TAKE_OUT")
    n_put = sum(1 for e in events if e["type"] == "PUT_BACK")
    count_str = f"Events: {len(events)} (TAKE_OUT: {n_take}, PUT_BACK: {n_put})"
    cv2.putText(display, count_str, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls hint (right side)
    cv2.putText(display, "[o] TAKE_OUT  [i] PUT_BACK  [SPACE] pause  [q] quit",
                (w - 520, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Draw tray ROI if defined
    if tray_roi:
        x, y, rw, rh = tray_roi
        cv2.rectangle(display, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
        cv2.putText(display, "TRAY ROI", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw recent event markers on a timeline bar
    bar_y = h - 30
    cv2.rectangle(display, (20, bar_y), (w - 20, bar_y + 10), (60, 60, 60), -1)
    for ev in events:
        frac = ev["timestamp_sec"] / max(total_sec, 1)
        bx = int(20 + frac * (w - 40))
        ev_color = (0, 0, 255) if ev["type"] == "TAKE_OUT" else (0, 255, 0)
        cv2.circle(display, (bx, bar_y + 5), 5, ev_color, -1)

    # Current position marker on timeline
    pos_frac = current_sec / max(total_sec, 1)
    pos_x = int(20 + pos_frac * (w - 40))
    cv2.drawMarker(display, (pos_x, bar_y + 5), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 12, 2)

    return display


def select_roi(frame: np.ndarray) -> list | None:
    """Let user select tray ROI interactively."""
    print("Select tray ROI. Press ENTER to confirm, C to cancel.")
    roi = cv2.selectROI("Select Tray ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Tray ROI")
    x, y, w, h = roi
    if w > 0 and h > 0:
        return [int(x), int(y), int(w), int(h)]
    return None


def save_annotations(output_path: Path, video_name: str,
                     events: list, tray_roi: list | None):
    """Write event annotations to JSON."""
    data = {
        "video": video_name,
        "tray_roi": tray_roi,
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(events)} events to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate tray events (TAKE_OUT / PUT_BACK) in a surgical video."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the source video file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the event annotation JSON"
    )
    parser.add_argument(
        "--load-existing", action="store_true",
        help="Load existing annotations from --output and continue editing"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.is_file():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    # Get video info
    vinfo = get_video_info(str(video_path))
    fps = vinfo["fps"]
    total_frames = vinfo["total_frames"]
    total_sec = vinfo["duration_s"]

    print(f"Video: {video_path.name}")
    print(f"Resolution: {vinfo['width']}x{vinfo['height']}, "
          f"{fps:.1f} fps, {total_frames} frames, {total_sec:.1f}s")
    print()
    print("Controls:")
    print("  [o] Mark TAKE_OUT    [i] Mark PUT_BACK")
    print("  [SPACE] Pause/Play   [LEFT/RIGHT] Seek +/-1s")
    print("  [+/-] Speed up/down  [r] Set tray ROI")
    print("  [u] Undo last event  [q/ESC] Quit & save")
    print()

    # Load existing annotations if requested
    events = []
    tray_roi = None
    if args.load_existing and output_path.is_file():
        with open(output_path) as f:
            existing = json.load(f)
        events = existing.get("events", [])
        tray_roi = existing.get("tray_roi")
        print(f"Loaded {len(events)} existing events")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    paused = False
    speed = 1.0
    window_name = f"Label Events - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
        else:
            # Re-read current frame when paused (for seeking)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            ret, frame = cap.read()
            if not ret:
                break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_sec = current_frame / fps

        info = {
            "current_sec": current_sec,
            "total_sec": total_sec,
            "current_frame": current_frame,
            "total_frames": total_frames,
        }

        display = draw_overlay(frame, info, events, paused, speed, tray_roi)
        cv2.imshow(window_name, display)

        # Compute wait time based on playback speed
        wait_ms = max(1, int((1000.0 / fps) / speed)) if not paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            break

        elif key == ord(" "):  # SPACE: pause/resume
            paused = not paused

        elif key == ord("o"):  # TAKE_OUT
            event = {
                "type": "TAKE_OUT",
                "timestamp_sec": round(current_sec, 2),
                "duration_sec": 2.0,
                "instrument_if_known": None,
                "hand_side": "unknown",
                "confidence": "certain",
            }
            events.append(event)
            print(f"  + TAKE_OUT at {current_sec:.2f}s (frame {current_frame})")

        elif key == ord("i"):  # PUT_BACK
            event = {
                "type": "PUT_BACK",
                "timestamp_sec": round(current_sec, 2),
                "duration_sec": 2.0,
                "instrument_if_known": None,
                "hand_side": "unknown",
                "confidence": "certain",
            }
            events.append(event)
            print(f"  + PUT_BACK at {current_sec:.2f}s (frame {current_frame})")

        elif key == 83 or key == ord("d"):  # RIGHT arrow or d: seek forward 1s
            new_frame = min(current_frame + int(fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

        elif key == 81 or key == ord("a"):  # LEFT arrow or a: seek backward 1s
            new_frame = max(current_frame - int(fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

        elif key == ord("+") or key == ord("="):  # Speed up
            speed = min(speed + 0.5, 8.0)
            print(f"  Speed: {speed:.1f}x")

        elif key == ord("-"):  # Speed down
            speed = max(speed - 0.5, 0.25)
            print(f"  Speed: {speed:.1f}x")

        elif key == ord("r"):  # Set ROI
            tray_roi = select_roi(frame)
            if tray_roi:
                print(f"  Tray ROI set: {tray_roi}")

        elif key == ord("u"):  # Undo last
            if events:
                removed = events.pop()
                print(f"  - Undid {removed['type']} at {removed['timestamp_sec']:.2f}s")
            else:
                print("  No events to undo.")

    cap.release()
    cv2.destroyAllWindows()

    # Sort events by timestamp before saving
    events.sort(key=lambda e: e["timestamp_sec"])

    save_annotations(output_path, video_path.name, events, tray_roi)


if __name__ == "__main__":
    main()
