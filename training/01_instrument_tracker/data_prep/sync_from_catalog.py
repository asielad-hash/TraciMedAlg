"""
Sync Instrument Data from SurgicalInstruments App
===================================================
Pulls instrument taxonomy, exemplar images, and kit definitions
from the SurgicalInstruments app API into the training data folder.

The SurgicalInstruments app is the SINGLE SOURCE OF TRUTH for:
  - Instrument names, families, manufacturers
  - Kit definitions (which instruments belong to which surgical kit)
  - Reference images of each instrument (exemplar library)
  - Expected instrument counts per kit

This script bridges the catalog app and the training pipeline.

Usage:
    # Pull everything (taxonomy + images + kits):
    python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data

    # Pull only taxonomy (no images):
    python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data --taxonomy-only

    # Pull from cloud deployment:
    python sync_from_catalog.py --api-url https://your-app.onrender.com --output-dir ../data

    # Update config.yaml taxonomy from catalog:
    python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data --update-config

Requires: SurgicalInstruments app running (locally or cloud).
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

# Shared library imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config


# -----------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------

def api_get(base_url: str, endpoint: str) -> dict:
    """GET request to the SurgicalInstruments API."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot reach {url}")
        print(f"  Is the SurgicalInstruments app running?")
        print(f"  Start it with: cd SurgicalInstruments && node server.js")
        raise SystemExit(1) from e


def download_image(base_url: str, filename: str, output_path: str) -> bool:
    """Download an image from the SurgicalInstruments app."""
    url = f"{base_url.rstrip('/')}/api/images/{filename}/data"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            # Check if it's a JSON response with base64 data
            try:
                json_data = json.loads(data)
                if "image_data" in json_data:
                    img_data = json_data["image_data"]
                    if img_data.startswith("data:"):
                        # Strip data URI prefix
                        img_data = img_data.split(",", 1)[1]
                    raw = base64.b64decode(img_data)
                    with open(output_path, "wb") as f:
                        f.write(raw)
                    return True
            except (json.JSONDecodeError, ValueError):
                pass
            # Raw image bytes
            with open(output_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"    WARNING: Failed to download {filename}: {e}")
        return False


# -----------------------------------------------------------------------
# Taxonomy sync
# -----------------------------------------------------------------------

def sync_taxonomy(base_url: str, output_dir: str) -> dict:
    """Pull instrument taxonomy from the catalog app.

    Creates:
      output_dir/taxonomy.json — full instrument list with families
      output_dir/families.json — family groupings
    """
    print("=" * 60)
    print("SYNCING INSTRUMENT TAXONOMY")
    print("=" * 60)

    # Get all instruments
    instruments = api_get(base_url, "/api/instruments")
    families_raw = api_get(base_url, "/api/families")
    stats = api_get(base_url, "/api/stats")

    print(f"  Instruments: {stats.get('instruments', len(instruments))}")
    print(f"  Families:    {stats.get('families', len(families_raw))}")
    print(f"  Images:      {stats.get('images', '?')}")
    print(f"  Videos:      {stats.get('videos', '?')}")

    # Build taxonomy organized by family
    taxonomy = defaultdict(list)
    instrument_list = []

    for inst in instruments:
        family = inst.get("family", "Unknown")
        entry = {
            "id": inst.get("id"),
            "name": inst.get("name", ""),
            "family": family,
            "manufacturer": inst.get("manufacturer", ""),
            "catalogue_number": inst.get("catalogueNumber", inst.get("catalogue_number", "")),
            "size_variant": inst.get("sizeVariant", inst.get("size_variant", "")),
            "image_count": inst.get("imageCount", inst.get("image_count", 0)),
            "video_count": inst.get("videoCount", inst.get("video_count", 0)),
            "kit_names": inst.get("kitNames", inst.get("kit_names", [])),
        }
        taxonomy[family].append(entry)
        instrument_list.append(entry)

    # Save taxonomy
    taxonomy_path = os.path.join(output_dir, "taxonomy.json")
    with open(taxonomy_path, "w") as f:
        json.dump({
            "total_instruments": len(instrument_list),
            "total_families": len(taxonomy),
            "families": dict(taxonomy),
            "instruments": instrument_list,
        }, f, indent=2)
    print(f"  Saved: {taxonomy_path}")

    # Save families summary
    families_path = os.path.join(output_dir, "families.json")
    families_summary = {}
    for family, insts in taxonomy.items():
        families_summary[family] = {
            "count": len(insts),
            "instruments": [i["name"] for i in insts],
            "with_images": sum(1 for i in insts if i["image_count"] > 0),
        }
    with open(families_path, "w") as f:
        json.dump(families_summary, f, indent=2)
    print(f"  Saved: {families_path}")

    return {"taxonomy": dict(taxonomy), "instruments": instrument_list}


# -----------------------------------------------------------------------
# Kit sync
# -----------------------------------------------------------------------

def sync_kits(base_url: str, output_dir: str) -> list:
    """Pull kit definitions from the catalog app.

    Creates:
      output_dir/kits/                    — one JSON per kit
      output_dir/kit_instrument_counts.json — expected counts per kit
    """
    print("\n" + "=" * 60)
    print("SYNCING KIT DEFINITIONS")
    print("=" * 60)

    kits_data = api_get(base_url, "/api/kit-catalogs")
    kits_dir = os.path.join(output_dir, "kits")
    os.makedirs(kits_dir, exist_ok=True)

    kit_counts = {}

    for kit in kits_data:
        kit_name = kit.get("name", "Unknown Kit")
        surgery_type = kit.get("surgery_type", kit.get("surgeryType", "unknown"))

        # Parse kit data
        kit_data_raw = kit.get("data", "{}")
        if isinstance(kit_data_raw, str):
            try:
                kit_data = json.loads(kit_data_raw)
            except json.JSONDecodeError:
                kit_data = {}
        else:
            kit_data = kit_data_raw

        # Extract instrument counts
        total_required = 0
        kit_instruments = []
        families = kit_data.get("families", [])

        for family in families:
            family_name = family.get("name", "Unknown")
            for inst in family.get("instruments", []):
                qty = inst.get("qty", 1)
                total_required += qty
                kit_instruments.append({
                    "name": inst.get("name", ""),
                    "family": family_name,
                    "qty": qty,
                    "manufacturer": inst.get("manufacturer", ""),
                    "product_number": inst.get("productNumber", ""),
                })

        kit_entry = {
            "name": kit_name,
            "surgery_type": surgery_type,
            "total_required": total_required,
            "instrument_count": len(kit_instruments),
            "instruments": kit_instruments,
        }

        # Save per-kit JSON
        safe_name = surgery_type.replace("/", "-").replace(" ", "_")
        kit_path = os.path.join(kits_dir, f"{safe_name}.json")
        with open(kit_path, "w") as f:
            json.dump(kit_entry, f, indent=2)

        kit_counts[kit_name] = {
            "total_required": total_required,
            "distinct_instruments": len(kit_instruments),
        }

        print(f"  {kit_name}: {total_required} instruments in {len(families)} families")

    # Save counts summary
    counts_path = os.path.join(output_dir, "kit_instrument_counts.json")
    with open(counts_path, "w") as f:
        json.dump(kit_counts, f, indent=2)
    print(f"\n  Saved: {counts_path}")
    print(f"  Kit JSONs: {kits_dir}/")

    return kits_data


# -----------------------------------------------------------------------
# Exemplar image sync
# -----------------------------------------------------------------------

def sync_exemplar_images(base_url: str, output_dir: str,
                          instruments: list, max_per_class: int = 100) -> dict:
    """Download instrument images as exemplar library.

    Creates:
      output_dir/exemplars/{family_name}/{instrument_name}/
        image_001.jpg, image_002.jpg, ...
    """
    print("\n" + "=" * 60)
    print("SYNCING EXEMPLAR IMAGES")
    print("=" * 60)

    exemplars_dir = os.path.join(output_dir, "exemplars")
    os.makedirs(exemplars_dir, exist_ok=True)

    stats = {"total_downloaded": 0, "total_skipped": 0, "per_class": {}}

    # Only download for instruments that have images
    instruments_with_images = [i for i in instruments if i["image_count"] > 0]
    print(f"  Instruments with images: {len(instruments_with_images)} / {len(instruments)}")

    for inst in instruments_with_images:
        inst_id = inst["id"]
        family = inst["family"].replace("/", "-").replace(" ", "_")
        name = inst["name"][:80].replace("/", "-").replace(" ", "_").replace(",", "")

        class_dir = os.path.join(exemplars_dir, family, name)
        os.makedirs(class_dir, exist_ok=True)

        # Get full instrument details with images
        try:
            detail = api_get(base_url, f"/api/instruments/{inst_id}")
        except SystemExit:
            continue

        images = detail.get("images", [])
        downloaded = 0

        for img in images[:max_per_class]:
            filename = img.get("filename", "")
            mime = img.get("mime_type", "")

            # Skip videos — only want still images for exemplars
            if "video" in mime:
                continue

            out_path = os.path.join(class_dir, f"exemplar_{downloaded + 1:03d}.jpg")
            if os.path.exists(out_path):
                downloaded += 1
                stats["total_skipped"] += 1
                continue

            if download_image(base_url, filename, out_path):
                downloaded += 1
                stats["total_downloaded"] += 1

        if downloaded > 0:
            print(f"  {inst['family']}/{inst['name'][:40]}: {downloaded} images")
            stats["per_class"][inst["name"]] = downloaded

    # Save stats
    stats_path = os.path.join(exemplars_dir, "sync_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Total downloaded: {stats['total_downloaded']}")
    print(f"  Total skipped (already exist): {stats['total_skipped']}")
    print(f"  Classes with images: {len(stats['per_class'])}")

    # Report classes with no images
    no_images = [i for i in instruments if i["image_count"] == 0]
    if no_images:
        print(f"\n  WARNING: {len(no_images)} instruments have NO images:")
        for i in no_images[:10]:
            print(f"    - {i['family']}: {i['name']}")
        if len(no_images) > 10:
            print(f"    ... and {len(no_images) - 10} more")

    return stats


# -----------------------------------------------------------------------
# Config update
# -----------------------------------------------------------------------

def update_config_taxonomy(config_path: str, taxonomy: dict):
    """Update config.yaml instrument taxonomy from catalog data.

    Groups instruments into priority tiers based on family:
      critical: sponges, needles, suture
      high: scissors, clamps, forceps
      medium: retractors, hemostatic, specialty
    """
    import yaml

    CRITICAL_FAMILIES = {"sponges", "gauze", "needles", "suture", "sharps"}
    HIGH_FAMILIES = {"scissors", "clamps", "forceps", "needle holders",
                     "cutting", "grasping", "clamping", "knife"}
    # Everything else is medium

    new_taxonomy = {"critical": [], "high": [], "medium": []}

    for family, instruments in taxonomy.items():
        family_lower = family.lower()
        for inst in instruments:
            name = inst["name"].lower()

            # Determine priority
            if any(kw in family_lower or kw in name for kw in CRITICAL_FAMILIES):
                tier = "critical"
            elif any(kw in family_lower or kw in name for kw in HIGH_FAMILIES):
                tier = "high"
            else:
                tier = "medium"

            new_taxonomy[tier].append(inst["name"])

    # Load and update config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["dataset"]["instrument_taxonomy"] = new_taxonomy

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Updated config taxonomy:")
    print(f"    Critical: {len(new_taxonomy['critical'])} instruments")
    print(f"    High:     {len(new_taxonomy['high'])} instruments")
    print(f"    Medium:   {len(new_taxonomy['medium'])} instruments")
    print(f"  Saved: {config_path}")


# -----------------------------------------------------------------------
# Mapping file for detection → kit validation
# -----------------------------------------------------------------------

def generate_detection_to_kit_mapping(output_dir: str, taxonomy: dict, kits_dir: str):
    """Generate mapping from detected instrument names to kit membership.

    This is used at inference time: when the detector identifies an
    instrument, we look up which kits it belongs to and verify the
    expected count.

    Creates: output_dir/detection_kit_mapping.json
    """
    print("\n" + "=" * 60)
    print("GENERATING DETECTION → KIT MAPPING")
    print("=" * 60)

    mapping = {}

    for family, instruments in taxonomy.items():
        for inst in instruments:
            name = inst["name"]
            kits = inst.get("kit_names", [])
            if kits:
                mapping[name] = {
                    "family": family,
                    "kits": kits,
                    "manufacturer": inst.get("manufacturer", ""),
                    "catalogue_number": inst.get("catalogue_number", ""),
                }

    # Also load kit-level counts
    kit_counts = {}
    for kit_file in Path(kits_dir).glob("*.json"):
        with open(kit_file) as f:
            kit = json.load(f)
        kit_name = kit["name"]
        for inst in kit.get("instruments", []):
            key = inst["name"]
            if key not in kit_counts:
                kit_counts[key] = {}
            kit_counts[key][kit_name] = inst.get("qty", 1)

    # Merge
    for name, info in mapping.items():
        info["expected_counts"] = kit_counts.get(name, {})

    out_path = os.path.join(output_dir, "detection_kit_mapping.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"  Mapped {len(mapping)} instruments to kits")
    print(f"  Saved: {out_path}")

    return mapping


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync instrument data from SurgicalInstruments app into training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sync (taxonomy + images + kits):
  python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data

  # Taxonomy only (fast, no image download):
  python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data --taxonomy-only

  # Update config.yaml with latest taxonomy:
  python sync_from_catalog.py --api-url http://localhost:3000 --output-dir ../data --update-config
        """,
    )
    parser.add_argument("--api-url", default="http://localhost:3000",
                        help="SurgicalInstruments app URL (default: http://localhost:3000)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (typically 01_instrument_tracker/data)")
    parser.add_argument("--taxonomy-only", action="store_true",
                        help="Only sync taxonomy, skip image download")
    parser.add_argument("--update-config", action="store_true",
                        help="Update config.yaml taxonomy from catalog")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (auto-detected if not given)")
    parser.add_argument("--max-images-per-class", type=int, default=100,
                        help="Max exemplar images per instrument class")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SURGICAL INSTRUMENTS CATALOG → TRAINING DATA SYNC")
    print(f"Source: {args.api_url}")
    print(f"Target: {args.output_dir}")
    print("=" * 60)

    # 1. Sync taxonomy
    result = sync_taxonomy(args.api_url, args.output_dir)

    # 2. Sync kit definitions
    sync_kits(args.api_url, args.output_dir)

    # 3. Sync exemplar images (unless taxonomy-only)
    if not args.taxonomy_only:
        sync_exemplar_images(
            args.api_url, args.output_dir,
            result["instruments"],
            max_per_class=args.max_images_per_class,
        )
    else:
        print("\n  Skipping image download (--taxonomy-only)")

    # 4. Generate detection → kit mapping
    kits_dir = os.path.join(args.output_dir, "kits")
    if os.path.exists(kits_dir):
        generate_detection_to_kit_mapping(
            args.output_dir, result["taxonomy"], kits_dir
        )

    # 5. Update config taxonomy (if requested)
    if args.update_config:
        config_path = args.config
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
        if os.path.exists(config_path):
            update_config_taxonomy(config_path, result["taxonomy"])
        else:
            print(f"\n  WARNING: Config not found at {config_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"  Taxonomy:  {args.output_dir}/taxonomy.json")
    print(f"  Families:  {args.output_dir}/families.json")
    print(f"  Kits:      {args.output_dir}/kits/")
    print(f"  Kit counts:{args.output_dir}/kit_instrument_counts.json")
    print(f"  Mapping:   {args.output_dir}/detection_kit_mapping.json")
    if not args.taxonomy_only:
        print(f"  Exemplars: {args.output_dir}/exemplars/")
    print()
    print("Next steps:")
    print("  1. Review taxonomy.json — verify instrument names match your data")
    print("  2. Check exemplars/ — ensure enough images per class (target: 50+)")
    print("  3. Run: python data_prep/extract_frames.py ...")
    print("  4. Run: python data_prep/generate_zero_shot.py ...")


if __name__ == "__main__":
    main()
