"""
Event detection — LOST/NEW/BACK state machine for instrument counting.
Extracted from trackInstruments/python/alert_overlay.py
"""

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
LOST_THRESHOLD = 5       # consecutive absent frames before "lost" fires
NEW_THRESHOLD = 5        # consecutive present frames before "new" fires
ALERT_DURATION_S = 2.0   # how long the banner stays on screen (seconds)


def _raw_instrument_count(visible_ids, instruments):
    """Sum physical instrument count for visible objects."""
    total = 0
    for oid in visible_ids:
        info = instruments.get(str(oid), {})
        total += info.get("count", 1)
    return total


def compute_stable_counts(meta: dict) -> list[int]:
    """Compute per-frame display count with 5-frame stability filter.

    Returns list of ints, one per frame. The displayed count only changes
    when the raw count has been at a new value for LOST_THRESHOLD consecutive
    frames.
    """
    frames = meta["frames"]
    instruments = meta["instruments"]
    threshold = max(LOST_THRESHOLD, NEW_THRESHOLD)

    if not frames:
        return []

    raw_counts = [
        _raw_instrument_count(f["obj_ids"], instruments) for f in frames
    ]

    display_counts = []
    display = raw_counts[0]
    pending = raw_counts[0]
    streak = 1

    for raw in raw_counts:
        if raw == pending:
            streak += 1
        else:
            pending = raw
            streak = 1
        if streak >= threshold:
            display = pending
        display_counts.append(display)

    return display_counts


def _short_label(text):
    """Shorten 'Instrument #N' to just '#N'."""
    if text.lower().startswith("instrument"):
        if "#" in text:
            return "#" + text.split("#")[-1]
    return text


def detect_events(meta: dict) -> list[dict]:
    """Walk frame metadata and return a list of events.

    Each event: {"type": "lost"|"new"|"back", "obj_id": int, "frame": int, "text": str}

    State machine per object:
      - Starts as "unseen"
      - First appearance → "visible" (if not in first frame, can fire "new")
      - Missing for LOST_THRESHOLD frames → fires "lost"
      - Re-appears for NEW_THRESHOLD frames after "lost" → fires "back"
    """
    frames = meta["frames"]
    instruments = meta["instruments"]

    all_ids = {int(k) for k in instruments}

    absent_streak = {oid: 0 for oid in all_ids}
    present_streak = {oid: 0 for oid in all_ids}
    total_seen = {oid: 0 for oid in all_ids}
    ever_seen = set()
    fired_lost = set()
    fired_new = set()

    first_visible = set(frames[0]["obj_ids"]) if frames else set()
    ever_seen.update(first_visible)

    events = []

    for fdata in frames:
        fidx = fdata["frame"]
        visible = set(fdata["obj_ids"])

        for oid in all_ids:
            info = instruments.get(str(oid), {})
            label = _short_label(info.get("text", f"obj_{oid}"))

            if oid in visible:
                ever_seen.add(oid)
                present_streak[oid] = present_streak.get(oid, 0) + 1
                total_seen[oid] = total_seen.get(oid, 0) + 1
                absent_streak[oid] = 0

                if (oid in fired_lost
                        and present_streak[oid] >= NEW_THRESHOLD):
                    fired_lost.discard(oid)
                    events.append({
                        "type": "back", "obj_id": oid,
                        "frame": fidx, "text": f"BACK: {label}",
                    })
                elif (oid not in first_visible
                        and oid not in fired_new
                        and oid not in fired_lost
                        and present_streak[oid] >= NEW_THRESHOLD):
                    fired_new.add(oid)
                    events.append({
                        "type": "new", "obj_id": oid,
                        "frame": fidx, "text": f"NEW: {label}",
                    })
            else:
                if oid not in ever_seen:
                    continue
                absent_streak[oid] = absent_streak.get(oid, 0) + 1
                present_streak[oid] = 0

                if (oid not in fired_lost
                        and absent_streak[oid] >= LOST_THRESHOLD
                        and total_seen[oid] >= NEW_THRESHOLD):
                    fired_lost.add(oid)
                    events.append({
                        "type": "lost", "obj_id": oid,
                        "frame": fidx, "text": f"LOST: {label}",
                    })

    return events
