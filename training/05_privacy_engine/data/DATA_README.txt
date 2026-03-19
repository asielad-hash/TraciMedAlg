================================================================
05 — PRIVACY ENGINE
Face Blur + Text/Badge Masking + Anonymization
================================================================

WHAT THIS COMPONENT DOES:
  Real-time anonymization of all OR video and audio:
    - Detects and blurs ALL faces (even with surgical masks/caps)
    - Detects and masks text on badges, lanyards, monitors, labels
    - Scrambles voice to remove speaker identity
    - Encrypts everything (AES-256 at rest, TLS 1.3 in transit)

  HARD CONSTRAINT: Must complete in < 200ms per frame.
  Runs BEFORE any other processing — nothing passes without
  anonymization first.

  TYPE: Uses pre-trained models (no custom training needed).
  Only threshold tuning for OR-specific conditions.

================================================================
EXPECTED DATA FOLDER STRUCTURE
================================================================

data/
├── videos/                     ← Test videos for validation
│   ├── privacy_test_001.mp4    ← OR footage with known faces/badges
│   └── ...
│
└── test_frames/                ← Individual test frames
    ├── face_with_mask_001.jpg  ← face wearing surgical mask
    ├── face_with_cap_002.jpg   ← face with surgical cap
    ├── face_partial_003.jpg    ← partially occluded face
    ├── badge_scrubs_001.jpg    ← name badge on scrubs
    ├── badge_lanyard_002.jpg   ← lanyard ID card
    ├── monitor_screen_001.jpg  ← patient info on monitor
    ├── whiteboard_001.jpg      ← OR whiteboard with names
    └── no_face_negative_001.jpg ← instrument-only (no face)

================================================================
TEST DATA REQUIREMENTS
================================================================

  To validate the privacy engine, you need:

  FACE DETECTION TEST SET:
    - 100+ frames with faces in OR conditions
    - Include: surgical masks, caps, side profiles, overhead angles
    - Include: multiple faces in frame
    - Include: negative samples (no faces — just instruments/tables)
    - Ground truth: bounding boxes around all faces

  TEXT/BADGE DETECTION TEST SET:
    - 50+ frames with visible text/badges
    - Include: badges on scrubs, lanyard IDs, monitor screens
    - Include: whiteboards with patient names
    - Ground truth: bounding boxes around all PHI text

  LATENCY TEST SET:
    - 1000+ frames at full resolution (1920x1080)
    - Measured on target OR edge hardware (RTX 4090)

================================================================
VALIDATION FORMAT
================================================================

  Ground truth file: {frame_name}_gt.json

  {
    "frame": "face_with_mask_001.jpg",
    "faces": [
      {"bbox": [100, 50, 200, 200], "visible": "masked"},
      {"bbox": [500, 80, 180, 190], "visible": "partial"}
    ],
    "text_regions": [
      {"bbox": [300, 400, 150, 30], "type": "badge"},
      {"bbox": [800, 100, 400, 300], "type": "monitor"}
    ]
  }

================================================================
HOW TO RUN
================================================================

  No training — only evaluation:

  TEST FACE DETECTION:
    python eval/evaluate.py \
      --config config/config.yaml \
      --test-dir data/test_frames \
      --mode faces

  TEST TEXT DETECTION:
    python eval/evaluate.py \
      --config config/config.yaml \
      --test-dir data/test_frames \
      --mode text

  TEST LATENCY:
    python eval/evaluate.py \
      --config config/config.yaml \
      --test-dir data/test_frames \
      --mode latency

  FULL VALIDATION (all):
    python eval/evaluate.py \
      --config config/config.yaml \
      --test-dir data/test_frames \
      --mode all

================================================================
COMPLIANCE CHECKLIST
================================================================

  [ ] Face detection recall >= 99%
  [ ] Badge/text detection recall >= 98%
  [ ] Latency < 200ms per frame on target hardware
  [ ] False positive rate < 5% (don't over-blur instruments)
  [ ] Voice scrambling preserves speech content
  [ ] AES-256 encryption verified
  [ ] Reviewed by site DPO (Data Protection Officer)
  [ ] HIPAA de-identification standard met
  [ ] No PHI persists on disk unencrypted at any point
================================================================
