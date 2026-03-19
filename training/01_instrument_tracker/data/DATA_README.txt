================================================================
01 — INSTRUMENT TRACKER
Detect, Track & Count Instruments on Mayo/Back Table
================================================================

WHAT THIS COMPONENT DOES:
  Instruments are clearly visible on the mayo or back table.
  The model detects each instrument, creates a pixel mask,
  tracks it across frames, classifies its type, and maintains
  a running count. Generates LOST/NEW/BACK alerts.

================================================================
EXPECTED DATA FOLDER STRUCTURE
================================================================

data/
├── videos/                     ← Source videos (anonymized)
│   ├── sheba_general_001.mp4
│   ├── kaiser_ortho_042.mp4
│   └── ...
│
├── annotations/                ← Instrument annotations
│   ├── coco_export/            ← COCO JSON from Label Studio / CVAT
│   │   ├── sheba_general_001.json
│   │   └── ...
│   └── zero_shot/              ← SAM3 auto-generated (needs review)
│       ├── sheba_general_001_tracked_meta.json
│       └── ...
│
├── exemplars/                  ← Instrument reference images
│   ├── surgical_sponge/        ← 50+ images per class
│   │   ├── sponge_clean_front.jpg
│   │   ├── sponge_used_side.jpg
│   │   └── ...
│   ├── scalpel/
│   ├── kelly_clamp/
│   ├── metzenbaum_scissors/
│   └── ...
│
├── images/                     ← Extracted frames (auto-generated)
│   └── (populated by data_prep script)
│
├── train.json                  ← (auto-generated)
├── val.json                    ← (auto-generated)
├── test.json                   ← (auto-generated)
└── split_metadata.json         ← (auto-generated)

================================================================
VIDEO NAMING CONVENTION
================================================================

  {site}_{specialty}_{case_id}.mp4

  Examples:
    sheba_general_001.mp4
    kaiser_ortho_042.mp4
    inova_cardiac_015.mp4
    northwestern_neuro_008.mp4

================================================================
ANNOTATION FORMAT (COCO JSON)
================================================================

  {
    "images": [
      {
        "id": 0,
        "file_name": "sheba_general_001_frame000000.jpg",
        "width": 1920,
        "height": 1080
      }
    ],
    "annotations": [
      {
        "id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "segmentation": [[x1,y1, x2,y2, x3,y3, ...]],
        "area": 1234
      }
    ],
    "categories": [
      {"id": 1, "name": "surgical sponge", "priority": "critical"},
      {"id": 2, "name": "scalpel", "priority": "high"}
    ]
  }

================================================================
ANNOTATION WORKFLOW
================================================================

  1. Run SAM3 zero-shot on video → auto-generates candidate masks
  2. Clinical annotator reviews in Label Studio / CVAT
  3. Corrects missed instruments, fixes wrong labels
  4. Exports to COCO JSON → annotations/coco_export/

  Expected time: ~30-60 min per case (with zero-shot pre-annotation)

================================================================
EXEMPLAR LIBRARY
================================================================

  Per instrument class:
    - 50+ images minimum
    - Clean crop, white background
    - Multiple orientations (front, side, angled)
    - Clean + used states
    - Sources: OR recordings, sterile processing dept, catalogs

================================================================
HOW TO RUN
================================================================

  PREPARE:
    python data_prep/prepare_data.py \
      --config config/config.yaml \
      --video-dir data/videos \
      --anno-dir data/annotations/coco_export \
      --output-dir data

  TRAIN:
    python train/train.py --config config/config.yaml --data-dir data

  EVALUATE:
    python eval/evaluate.py --config config/config.yaml --data-dir data

================================================================
LEGACY (V1 REFERENCE)
================================================================

  The legacy/ folder contains the V1 prototype of this component,
  built on GroundingDINO + SAM2 (before SAM3 unified the pipeline):

  legacy/
  ├── tool_tracking.py        ← V1 class: GroundingDINO detection
  │                              + SAM2 segmentation + propagation
  └── track_grounded.ipynb    ← V1 notebook: original experiment
                                 (demo 7.mp4, 16 instruments detected)

  V1 pipeline (3 separate models):
    GroundingDINO → bounding boxes → SAM2 → masks → SAM2 propagation

  Current pipeline (unified):
    SAM3 → detection + segmentation + tracking in single pass

  V1 is kept for reference only. All new training uses SAM3.

================================================================
ACCEPTANCE CRITERIA
================================================================

  Detection AUC-ROC:    >= 0.90 per instrument class
  Segmentation mIoU:    >= 0.85
  Tracking HOTA:        >= 0.80
  Count accuracy:       >= 95% per procedure
  False alarm rate:     < 5%
  Latency:              median < 1s, p95 < 1.5s
================================================================
