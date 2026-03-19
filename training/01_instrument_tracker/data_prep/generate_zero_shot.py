"""
Run SAM3 zero-shot detection and tracking on surgical videos.

For each video, starts a SAM3 session, sends text prompts for every
instrument in the config taxonomy, runs bidirectional propagation,
applies identity-swap correction, and saves _tracked_meta.json.

NOTE: Must run in the sam3 conda environment.

Usage:
    conda activate sam3
    python generate_zero_shot.py --config ../config/config.yaml --video-dir /path/to/videos
    python generate_zero_shot.py --video-dir /videos --output-dir ./data/tracked --ckpt sam3.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.sam3_utils import load_predictor, start_session, propagate_full, close_session
from shared.video_io import get_video_info
from shared.mask_utils import fix_identity_swaps, mask_to_bbox, mask_area

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_instrument_prompts(config: dict) -> list[dict]:
    """Build obj_id-indexed text prompts from the config instrument taxonomy."""
    taxonomy = config.get("dataset", {}).get("instrument_taxonomy", {})
    prompts = []
    obj_id = 1
    for priority, instruments in taxonomy.items():
        for name in instruments:
            prompts.append({"obj_id": obj_id, "text": name, "priority": priority})
            obj_id += 1
    return prompts


def build_tracked_meta(video_info: dict, prompts: list[dict],
                       outputs: dict) -> dict:
    """Convert raw SAM3 propagation outputs into _tracked_meta.json format."""
    obj_map = {p["obj_id"]: p for p in prompts}

    frames = []
    for fidx in sorted(outputs.keys()):
        out = outputs[fidx]
        obj_ids = out.get("out_obj_ids")
        bmasks = out.get("out_binary_masks")
        if obj_ids is None or bmasks is None:
            continue

        frame_data = {"frame": int(fidx), "obj_ids": [], "bboxes": [],
                      "areas": [], "labels": []}

        for i, oid in enumerate(obj_ids):
            oid_int = int(oid)
            mask = bmasks[i]
            if hasattr(mask, "squeeze"):
                mask = mask.squeeze()

            bbox = mask_to_bbox(mask)
            area = mask_area(mask)
            if bbox is None or area < 10:
                continue

            info = obj_map.get(oid_int, {})
            frame_data["obj_ids"].append(oid_int)
            frame_data["bboxes"].append(list(bbox))
            frame_data["areas"].append(area)
            frame_data["labels"].append(info.get("text", f"object_{oid_int}"))

        if frame_data["obj_ids"]:
            frames.append(frame_data)

    return {
        "video": video_info["name"],
        "video_stem": video_info["stem"],
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
        "total_frames": video_info["total_frames"],
        "instruments": [
            {"obj_id": p["obj_id"], "text": p["text"], "priority": p["priority"]}
            for p in prompts
        ],
        "num_tracked_frames": len(frames),
        "frames": frames,
    }


def process_video(predictor, video_path: Path, prompts: list[dict],
                  output_dir: Path) -> dict:
    """Run zero-shot SAM3 tracking on a single video and save results."""
    import torch

    video_info = get_video_info(str(video_path))
    print(f"  {video_info['width']}x{video_info['height']}, "
          f"{video_info['total_frames']} frames, {video_info['fps']:.1f} fps")

    session_id = start_session(predictor, str(video_path))
    print(f"  Session: {session_id}")

    try:
        # Add text prompts on frame 0
        print(f"  Adding {len(prompts)} text prompts ...")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt["text"],
                    "obj_id": prompt["obj_id"],
                    "fill_hole_area": 0,
                })

        # Propagate
        print("  Propagating ...")
        outputs = propagate_full(
            predictor, session_id,
            total_frames=video_info["total_frames"],
            keyframe_indices=[0],
        )

        # Fix identity swaps
        outputs = fix_identity_swaps(outputs, video_info["total_frames"])

    finally:
        close_session(predictor, session_id)

    meta = build_tracked_meta(video_info, prompts, outputs)
    out_path = output_dir / f"{video_info['stem']}_tracked_meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {out_path} ({meta['num_tracked_frames']} frames)")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 zero-shot detection on surgical videos. "
                    "Must run in sam3 conda environment."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).resolve().parent.parent / "config" / "config.yaml"),
        help="Path to config YAML (default: ../config/config.yaml)",
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing source video files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for _tracked_meta.json (default: data/tracked/).",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to SAM3 checkpoint (default: auto-detect).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip videos whose _tracked_meta.json already exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["_config_dir"]).parent / "data" / "tracked"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_instrument_prompts(config)
    print(f"Taxonomy: {len(prompts)} instrument categories")
    for p in prompts:
        print(f"  [{p['priority']}] {p['text']}")
    print()

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    if args.skip_existing:
        videos = [
            v for v in videos
            if not (output_dir / f"{v.stem}_tracked_meta.json").exists()
        ]
        print(f"After skipping existing: {len(videos)} to process")

    print(f"Processing {len(videos)} video(s) -> {output_dir}\n")

    predictor = load_predictor(ckpt_path=args.ckpt)

    t_total = time.time()
    results = []

    for i, vpath in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vpath.name}")
        t0 = time.time()
        try:
            process_video(predictor, vpath, prompts, output_dir)
            results.append({"video": vpath.name, "ok": True, "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"video": vpath.name, "ok": False, "error": str(e)})
        print()

    elapsed = round(time.time() - t_total, 1)
    ok = sum(1 for r in results if r["ok"])
    print(f"Done. {ok}/{len(results)} succeeded in {elapsed}s.")

    summary_path = output_dir / "zero_shot_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "elapsed_s": elapsed}, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
