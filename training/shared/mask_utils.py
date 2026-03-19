"""
Mask utilities — rendering, identity swap correction, geometry.
Extracted from trackInstruments/python/track.py
"""

import cv2
import numpy as np


# 8 distinct colors for up to 8 instruments; cycles if more
COLORS_RGB = [
    (220, 50,  50),   (50,  200, 50),   (50,  100, 230),  (230, 140, 20),
    (160, 50,  200),  (20,  190, 190),  (210, 190, 20),   (200, 80,  150),
]


def rgb_to_bgr(rgb):
    """Convert RGB tuple to BGR tuple."""
    return (rgb[2], rgb[1], rgb[0])


def color_to_bgr(color):
    """Convert a color (hex string or RGB tuple/list) to a BGR tuple."""
    if isinstance(color, str):
        c = color.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return (b, g, r)
    return rgb_to_bgr(tuple(color))


def mask_centroid(mask: np.ndarray):
    """Return (cx, cy) centroid of a binary mask, or None if empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_to_bbox(mask: np.ndarray) -> tuple:
    """Return (x, y, w, h) bounding box of a binary mask, or None if empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return (x1, y1, x2 - x1, y2 - y1)


def mask_area(mask: np.ndarray) -> int:
    """Return number of nonzero pixels in mask."""
    return int((mask > 0).sum())


def overlay_masks(frame_bgr, masks: dict, ann_map: dict, alpha: float = 0.35):
    """Draw colored semi-transparent masks + contours + labels onto a BGR frame.

    Args:
        frame_bgr: HxWx3 uint8 BGR image
        masks: {obj_id: HxW uint8 binary mask}
        ann_map: {obj_id: {color, text, count}} per object
        alpha: opacity of the color fill

    Returns: new BGR frame with overlays
    """
    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    combined = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    for oid, mask in masks.items():
        if mask.sum() == 0:
            continue
        ann = ann_map.get(oid)
        if ann is None:
            continue
        count = ann.get("count", 1)
        multi = count > 1
        cbgr = np.array(color_to_bgr(ann["color"]), dtype=np.uint8)

        color_layer[mask > 0] = cbgr
        combined[mask > 0] = 1

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_px = 4 if multi else 2
        cv2.drawContours(overlay, cnts, -1, cbgr.tolist(), contour_px)
        if multi:
            cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)

        if cnts:
            c = mask_centroid(mask)
            if c:
                cx, cy = int(c[0]), int(c[1])
                lbl = ann.get("text", f"#{oid}")
                short = lbl.split("(")[0].strip() if "(" in lbl else lbl
                if short.lower().startswith("instrument"):
                    short = "#" + short.split("#")[-1] if "#" in short else short
                cv2.putText(overlay, short, (cx - 30, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cbgr.tolist(), 2)
                cv2.putText(overlay, short, (cx - 30, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if multi:
                    badge = f"x{count}"
                    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    bx, by = cx - bw // 2, cy + 18
                    cv2.rectangle(overlay, (bx - 4, by - bh - 4),
                                  (bx + bw + 4, by + 6), cbgr.tolist(), -1)
                    cv2.putText(overlay, badge, (bx, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mr = combined > 0
    if mr.any():
        overlay[mr] = cv2.addWeighted(overlay[mr], 1 - alpha, color_layer[mr], alpha, 0)
    return overlay


def fix_identity_swaps(outputs_per_frame: dict, total: int) -> dict:
    """Post-process SAM3 propagation output to fix identity swaps.

    After occlusion, SAM3 can swap which mask belongs to which obj_id.
    Detects swaps by tracking centroids and corrects them.
    """
    if not outputs_per_frame:
        return outputs_per_frame

    prev_centroids = {}
    remap = {}
    swap_count = 0

    for fidx in range(total):
        out = outputs_per_frame.get(fidx)
        if out is None:
            continue

        obj_ids = out.get("out_obj_ids")
        bmasks = out.get("out_binary_masks")
        if obj_ids is None or bmasks is None:
            continue

        # Apply current remap
        if remap:
            new_ids = []
            for oid in obj_ids:
                oid_int = int(oid)
                new_ids.append(remap.get(oid_int, oid_int))
            out["out_obj_ids"] = np.array(new_ids, dtype=obj_ids.dtype)
            obj_ids = out["out_obj_ids"]

        # Compute centroids
        curr_centroids = {}
        for i, oid in enumerate(obj_ids):
            oid = int(oid)
            m = bmasks[i]
            if hasattr(m, "squeeze"):
                m = m.squeeze()
            c = mask_centroid(m if not hasattr(m, 'numpy') else m)
            if c is not None:
                curr_centroids[oid] = c

        # Check for pairwise swaps
        if prev_centroids and curr_centroids:
            curr_ids = list(curr_centroids.keys())
            for i_a in range(len(curr_ids)):
                for i_b in range(i_a + 1, len(curr_ids)):
                    id_a, id_b = curr_ids[i_a], curr_ids[i_b]
                    if id_a not in prev_centroids or id_b not in prev_centroids:
                        continue

                    prev_a, prev_b = prev_centroids[id_a], prev_centroids[id_b]
                    curr_a, curr_b = curr_centroids[id_a], curr_centroids[id_b]

                    dist_a = np.hypot(curr_a[0] - prev_a[0], curr_a[1] - prev_a[1])
                    dist_b = np.hypot(curr_b[0] - prev_b[0], curr_b[1] - prev_b[1])
                    dist_a2b = np.hypot(curr_a[0] - prev_b[0], curr_a[1] - prev_b[1])
                    dist_b2a = np.hypot(curr_b[0] - prev_a[0], curr_b[1] - prev_a[1])

                    min_jump = 50
                    if (dist_a > min_jump and dist_b > min_jump
                            and dist_a2b < dist_a * 0.5
                            and dist_b2a < dist_b * 0.5):
                        swap_count += 1
                        idx_a = list(obj_ids).index(id_a)
                        idx_b = list(obj_ids).index(id_b)
                        bmasks[[idx_a, idx_b]] = bmasks[[idx_b, idx_a]]

                        rev_a, rev_b = id_a, id_b
                        for orig, mapped in remap.items():
                            if mapped == id_a:
                                rev_a = orig
                            if mapped == id_b:
                                rev_b = orig
                        remap[rev_a] = id_b
                        remap[rev_b] = id_a
                        curr_centroids[id_a], curr_centroids[id_b] = (
                            curr_centroids[id_b], curr_centroids[id_a])

        prev_centroids = curr_centroids

    if swap_count:
        print(f"  [Identity fix] Corrected {swap_count} swap(s)")
    return outputs_per_frame
