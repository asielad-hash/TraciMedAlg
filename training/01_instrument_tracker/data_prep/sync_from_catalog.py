"""
Sync Instrument Data from SurgicalInstruments App
===================================================
Pulls instrument taxonomy, exemplar images/videos, and kit definitions
from the SurgicalInstruments app API into the training data folder.

Uses a sync_manifest.json to track what's already downloaded — only
NEW or UPDATED instruments are re-downloaded on subsequent runs.

Usage:
    python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data
"""

import argparse
import json
import os
import re
import sys
import random
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Shared library imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

VIDEO_EXTS = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
FRAMES_PER_VIDEO = 10


# ── Helpers ──────────────────────────────────────────────────

def api_get(base_url, endpoint):
    """GET JSON from the SurgicalInstruments API."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot reach {url}")
        print(f"  Is the SurgicalInstruments app running?")
        raise SystemExit(1) from e


def download_binary(base_url, filename):
    """Download binary image/video data from the API. Returns bytes or None."""
    url = f"{base_url.rstrip('/')}/api/images/{filename}/data"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        if len(data) < 100:
            return None
        return data
    except Exception:
        return None


def safe_dirname(name):
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:80]


def progress_bar(current, total, prefix='', width=40):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r  {prefix} [{bar}] {current}/{total} ({pct:.0%})')
    sys.stdout.flush()


def extract_video_frames(video_path, output_dir, stem, n_frames=10):
    """Extract random frames from a video file. Returns count of frames saved."""
    if not HAS_CV2:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return 0
    n = min(n_frames, frame_count)
    indices = sorted(random.sample(range(frame_count), n))
    saved = 0
    for fi, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, f"{stem}_frame{fi:03d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
    cap.release()
    return saved


def verify_file(path):
    """Check if a file is a valid image or video. Returns (type, ok)."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        if HAS_CV2:
            img = cv2.imread(str(path))
            return ('image', img is not None and img.shape[0] > 10)
        return ('image', os.path.getsize(path) > 100)
    elif ext in VIDEO_EXTS:
        if HAS_CV2:
            cap = cv2.VideoCapture(str(path))
            ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
            return ('video', ok)
        return ('video', os.path.getsize(path) > 100)
    return ('unknown', False)


# ── Sync functions (callable from notebook) ──────────────────

def sync_taxonomy(base_url, output_dir):
    """Pull instruments, families, kits from catalog. Returns (instruments, kits, families)."""
    instruments = api_get(base_url, "/api/instruments")
    kits = api_get(base_url, "/api/kit-catalogs")
    families = api_get(base_url, "/api/families")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "kits"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "exemplars"), exist_ok=True)

    with open(os.path.join(output_dir, "taxonomy.json"), "w") as f:
        json.dump(instruments, f, indent=2)
    with open(os.path.join(output_dir, "families.json"), "w") as f:
        json.dump(families, f, indent=2)
    for kit in kits:
        with open(os.path.join(output_dir, "kits", f"kit_{kit.get('id','x')}.json"), "w") as f:
            json.dump(kit, f, indent=2)

    return instruments, kits, families


def plan_sync(instruments, manifest_path):
    """Compare catalog vs manifest. Returns (to_download, to_skip, manifest)."""
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"instruments": {}}

    prev = manifest.get("instruments", {})
    to_download = []
    to_skip = []

    for inst in instruments:
        iid = str(inst['id'])
        img_count = inst.get('imageCount', inst.get('image_count', 0))
        updated = inst.get('updatedAt', inst.get('updated_at', ''))

        if iid not in prev:
            to_download.append((inst, "NEW"))
        elif prev[iid].get('imageCount', 0) != img_count:
            to_download.append((inst, f"UPDATED ({prev[iid].get('imageCount',0)}->{img_count} images)"))
        elif prev[iid].get('updatedAt', '') != updated:
            to_download.append((inst, "MODIFIED"))
        else:
            to_skip.append(inst)

    return to_download, to_skip, manifest


def download_instruments(base_url, output_dir, to_download, manifest):
    """Download images+videos for instruments in to_download list. Returns stats dict."""
    stats = {
        "images": 0, "videos": 0, "frames": 0,
        "skipped": 0, "errors": [], "per_instrument": {}
    }
    new_manifest = manifest.get("instruments", {}).copy()

    for idx, (inst, reason) in enumerate(to_download):
        progress_bar(idx + 1, len(to_download), prefix='Downloading')

        iid = str(inst['id'])
        name = inst.get('name', f"unknown_{iid}")
        inst_dir = os.path.join(output_dir, "exemplars", safe_dirname(name))
        os.makedirs(inst_dir, exist_ok=True)

        try:
            detail = api_get(base_url, f"/api/instruments/{iid}")
        except SystemExit:
            stats["errors"].append(f"{name}: API error")
            continue

        files = []
        for img in detail.get('images', []):
            filename = img.get('filename', '')
            if not filename:
                continue

            raw = download_binary(base_url, filename)
            if raw is None:
                stats["skipped"] += 1
                continue

            out_path = os.path.join(inst_dir, filename)
            with open(out_path, 'wb') as f:
                f.write(raw)

            ext = Path(filename).suffix.lower()
            if ext in VIDEO_EXTS:
                stats["videos"] += 1
                n_frames = extract_video_frames(
                    out_path, inst_dir, Path(filename).stem, FRAMES_PER_VIDEO)
                stats["frames"] += n_frames
                files.append({"filename": filename, "type": "video", "size": len(raw), "frames": n_frames})
            else:
                stats["images"] += 1
                files.append({"filename": filename, "type": "image", "size": len(raw)})

        new_manifest[iid] = {
            "name": name,
            "imageCount": inst.get('imageCount', inst.get('image_count', 0)),
            "videoCount": inst.get('videoCount', inst.get('video_count', 0)),
            "updatedAt": inst.get('updatedAt', inst.get('updated_at', '')),
            "files": files,
            "synced_at": datetime.now().isoformat()
        }
        stats["per_instrument"][name] = {"images": sum(1 for f in files if f["type"] == "image"),
                                          "videos": sum(1 for f in files if f["type"] == "video")}

    if to_download:
        print()  # newline after progress bar

    # Save manifest
    manifest_path = os.path.join(output_dir, "sync_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "last_sync": datetime.now().isoformat(),
            "api_url": base_url,
            "instruments": new_manifest
        }, f, indent=2)

    return stats


def verify_downloads(output_dir):
    """Verify all files in exemplars/. Returns stats dict."""
    stats = {"valid_images": 0, "valid_videos": 0, "corrupt": [], "empty_dirs": []}
    exemplars = os.path.join(output_dir, "exemplars")
    if not os.path.exists(exemplars):
        return stats

    dirs = sorted(Path(exemplars).iterdir())
    for d_idx, d in enumerate(dirs):
        if not d.is_dir():
            continue
        progress_bar(d_idx + 1, len(dirs), prefix='Verifying')
        files = list(d.iterdir())
        if not files:
            stats["empty_dirs"].append(d.name)
            continue
        for f in files:
            ftype, ok = verify_file(f)
            if ok:
                if ftype == 'image':
                    stats["valid_images"] += 1
                elif ftype == 'video':
                    stats["valid_videos"] += 1
            else:
                stats["corrupt"].append(str(f))
                f.unlink()

    if dirs:
        print()  # newline after progress bar
    return stats


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync from SurgicalInstruments catalog")
    parser.add_argument("--api-url", default="http://localhost:3000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--taxonomy-only", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded files")
    args = parser.parse_args()

    print(f"Syncing from {args.api_url} -> {args.output_dir}")

    instruments, kits, families = sync_taxonomy(args.api_url, args.output_dir)
    print(f"  Instruments: {len(instruments)}, Kits: {len(kits)}, Families: {len(families)}")

    if not args.taxonomy_only:
        manifest_path = os.path.join(args.output_dir, "sync_manifest.json")
        to_download, to_skip, manifest = plan_sync(instruments, manifest_path)
        print(f"  To download: {len(to_download)}, Skipped: {len(to_skip)}")

        if to_download:
            stats = download_instruments(args.api_url, args.output_dir, to_download, manifest)
            print(f"  Images: {stats['images']}, Videos: {stats['videos']}, Frames: {stats['frames']}")

    if args.verify:
        v = verify_downloads(args.output_dir)
        print(f"  Valid images: {v['valid_images']}, Videos: {v['valid_videos']}, Corrupt: {len(v['corrupt'])}")

    print("Done.")


if __name__ == "__main__":
    main()
