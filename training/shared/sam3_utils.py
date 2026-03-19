"""
SAM3 utilities — model loading, session management, propagation.
Extracted from trackInstruments/python/track.py and web/sam3_bridge.py

NOTE: Must run in the sam3 conda environment.
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# SAM3 model loading (lazy singleton)
# ---------------------------------------------------------------------------

_predictor = None


def load_predictor(ckpt_path: str = None):
    """Lazily load and return the SAM3 video predictor singleton.

    The model (~3.3 GB) is loaded on first call; subsequent calls
    return the cached instance.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    from sam3.model_builder import build_sam3_video_predictor

    if ckpt_path is None:
        # Default: look for sam3.pt relative to trackInstruments
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "trackInstruments" / "sam3.pt",
            Path(__file__).resolve().parent.parent / "sam3.pt",
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = str(c)
                break
        else:
            raise FileNotFoundError(
                "sam3.pt not found. Pass ckpt_path or place it in trackInstruments/")

    print(f"[SAM3] Loading model from {ckpt_path} ...")
    _predictor = build_sam3_video_predictor(checkpoint_path=ckpt_path)
    return _predictor


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def start_session(predictor, video_path: str) -> str:
    """Start a SAM3 session on a video. Returns session_id."""
    import torch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(video_path),
        })
    return resp["session_id"]


def close_session(predictor, session_id: str):
    """Close a SAM3 session and free GPU memory."""
    import torch
    predictor.handle_request({
        "type": "close_session",
        "session_id": session_id,
    })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_point_prompt(predictor, session_id: str, frame_idx: int,
                     points: list, labels: list, obj_id: int,
                     width: int, height: int) -> dict:
    """Send a point prompt to SAM3.

    Args:
        points: list of [x, y] in pixel coordinates
        labels: list of 1 (include) or 0 (exclude) per point
        width, height: frame dimensions (for normalization)

    Returns: SAM3 response dict with mask data
    """
    import torch

    # Normalize points to [0, 1] range
    norm_pts = [[x / width, y / height] for x, y in points]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "points": torch.tensor(norm_pts, dtype=torch.float32),
            "point_labels": torch.tensor(labels, dtype=torch.int32),
            "obj_id": obj_id,
            "fill_hole_area": 0,  # SAM3 bug: max_area=16 wipes small masks
        })
    return result


def propagate_full(predictor, session_id: str, total_frames: int,
                   keyframe_indices: list = None) -> dict:
    """Run forward + backward propagation and collect all frame outputs.

    Returns: {frame_idx: {"out_obj_ids": array, "out_binary_masks": array}}
    """
    import torch

    if keyframe_indices is None:
        keyframe_indices = [0]

    first_kf = min(keyframe_indices)
    last_kf = max(keyframe_indices)

    outputs = {}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward propagation
        for resp in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": first_kf,
        }):
            fidx = resp["frame_index"]
            out = resp["outputs"]
            obj_ids = out.get("out_obj_ids")
            bmasks = out.get("out_binary_masks")
            if obj_ids is not None and bmasks is not None:
                outputs[fidx] = {
                    "out_obj_ids": obj_ids.cpu().numpy() if hasattr(obj_ids, 'cpu') else np.array(obj_ids),
                    "out_binary_masks": bmasks.cpu().numpy() if hasattr(bmasks, 'cpu') else np.array(bmasks),
                }

        # Backward propagation (if keyframes aren't at frame 0)
        if first_kf > 0:
            for resp in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "backward",
                "start_frame_index": first_kf,
            }):
                fidx = resp["frame_index"]
                out = resp["outputs"]
                obj_ids = out.get("out_obj_ids")
                bmasks = out.get("out_binary_masks")
                if obj_ids is not None and bmasks is not None:
                    outputs[fidx] = {
                        "out_obj_ids": obj_ids.cpu().numpy() if hasattr(obj_ids, 'cpu') else np.array(obj_ids),
                        "out_binary_masks": bmasks.cpu().numpy() if hasattr(bmasks, 'cpu') else np.array(bmasks),
                    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Propagation complete: {len(outputs)} frames")
    return outputs
