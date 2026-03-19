================================================================
02 — TRAY MONITOR
Detect Instruments Taken Out / Put Back Into Crowded Tray
================================================================

WHAT THIS COMPONENT DOES:
  The instrument tray (back table) contains too many instruments
  packed together to individually detect and track each one.
  Instead, this component monitors for EVENTS:
    - TAKE_OUT: a hand reaches into the tray and removes something
    - PUT_BACK: a hand returns an instrument to the tray

  This is fundamentally different from Instrument Tracker (#01):
    Instrument Tracker = "I see each instrument, track them all"
    Tray Monitor  = "I can't see individual items, but I watch
                     for hands going in/out"

================================================================
EXPECTED DATA FOLDER STRUCTURE
================================================================

data/
├── videos/                     ← Source videos of tray (anonymized)
│   ├── sheba_general_001_tray.mp4
│   └── ...
│
├── annotations/                ← Event annotations
│   ├── sheba_general_001_tray_events.json
│   └── ...
│
├── event_clips/                ← Extracted event clips (auto-generated)
│   └── (populated by data_prep script)
│
├── train.json                  ← (auto-generated)
├── val.json                    ← (auto-generated)
└── test.json                   ← (auto-generated)

================================================================
EVENT ANNOTATION FORMAT
================================================================

  Per-video annotation: {video_stem}_events.json

  {
    "video": "sheba_general_001_tray.mp4",
    "tray_roi": [100, 50, 800, 600],
    "events": [
      {
        "type": "TAKE_OUT",
        "timestamp_sec": 45.2,
        "duration_sec": 2.1,
        "instrument_if_known": "kelly clamp",
        "hand_side": "right",
        "confidence": "certain"
      },
      {
        "type": "PUT_BACK",
        "timestamp_sec": 120.8,
        "duration_sec": 1.8,
        "instrument_if_known": "kelly clamp",
        "hand_side": "right",
        "confidence": "certain"
      },
      {
        "type": "TAKE_OUT",
        "timestamp_sec": 200.5,
        "duration_sec": 2.5,
        "instrument_if_known": null,
        "hand_side": "left",
        "confidence": "probable"
      }
    ]
  }

  FIELDS:
    type:                 "TAKE_OUT" | "PUT_BACK"
    timestamp_sec:        when the event starts
    duration_sec:         how long the hand is in the tray
    instrument_if_known:  instrument name (null if can't tell)
    hand_side:            "left" | "right" | "unknown"
    confidence:           "certain" | "probable" | "uncertain"

  TRAY ROI:
    [x, y, width, height] — bounding box of the tray area.
    Only events within this ROI are relevant.

================================================================
LABELING GUIDELINES
================================================================

  WHAT TO LABEL:
    - Every time a hand enters the tray and removes something
    - Every time a hand enters the tray and puts something back
    - If you can identify the instrument, note it (not required)

  WHAT NOT TO LABEL:
    - Hand hovering over tray without touching
    - Rearranging instruments within the tray (no IN/OUT)
    - Drape adjustment near the tray

  EDGE CASES:
    - Multiple instruments grabbed at once → single TAKE_OUT event,
      note "multiple" in instrument_if_known
    - Quick touch-and-release → label as TAKE_OUT if instrument
      was actually removed, skip if nothing was taken

================================================================
HOW THE DETECTION WORKS
================================================================

  Three signals are combined:

  1. HAND DETECTION
     Detect hand/glove entering the tray ROI boundary.
     Entry = potential event start. Exit = event end.

  2. FRAME DIFFERENCING
     Motion energy spike within the tray ROI indicates
     physical interaction with instruments.

  3. COUNT DELTA (when possible)
     If visible instrument count changes after the hand leaves,
     confirms TAKE_OUT (count decreased) or PUT_BACK (increased).

================================================================
HOW TO RUN
================================================================

  PREPARE:
    python data_prep/prepare_data.py \
      --config config/config.yaml \
      --video-dir data/videos \
      --anno-dir data/annotations \
      --output-dir data

  TRAIN:
    python train/train.py --config config/config.yaml --data-dir data

  EVALUATE:
    python eval/evaluate.py --config config/config.yaml --data-dir data

================================================================
ACCEPTANCE CRITERIA
================================================================

  Event detection F1:              >= 0.85
  Event classification accuracy:   >= 0.90 (TAKE_OUT vs PUT_BACK)
  False positive rate:             < 10%
  Latency:                         median < 500ms
================================================================
