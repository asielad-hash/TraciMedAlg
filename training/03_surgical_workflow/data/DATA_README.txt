================================================================
03 — SURGICAL WORKFLOW
Phase Recognition + Idle Time Detection
================================================================

WHAT THIS COMPONENT DOES:

  A) PHASE RECOGNITION
     Classifies each moment in surgery into one of 5 phases:
       0 - Setup / Preparation
       1 - Incision / Active Surgery
       2 - Count & Reconciliation
       3 - Closure
       4 - Turnover / Room Reset

  B) IDLE DETECTION
     Detects non-operative idle periods (no productive activity).
     Outputs idle time reports for hospital operations teams.

================================================================
EXPECTED DATA FOLDER STRUCTURE
================================================================

data/
├── videos/                     ← Source videos (anonymized)
│   ├── sheba_general_001.mp4
│   └── ...
│
├── phase_labels/               ← Phase boundary annotations
│   ├── sheba_general_001_phases.json
│   └── ...
│
├── idle_labels/                ← Idle period annotations
│   ├── sheba_general_001_idle.json
│   └── ...
│
├── clips/                      ← Extracted temporal clips (auto-generated)
│   └── (populated by data_prep script)
│
├── train.json                  ← (auto-generated)
├── val.json                    ← (auto-generated)
└── test.json                   ← (auto-generated)

================================================================
PHASE LABEL FORMAT
================================================================

  File: {video_stem}_phases.json

  {
    "video": "sheba_general_001.mp4",
    "annotator": "Dr. Cohen",
    "phases": [
      {
        "phase": "setup_preparation",
        "start_sec": 0,
        "end_sec": 120,
        "notes": "patient positioning, draping"
      },
      {
        "phase": "incision_active_surgery",
        "start_sec": 120,
        "end_sec": 3600,
        "notes": ""
      },
      {
        "phase": "count_reconciliation",
        "start_sec": 3600,
        "end_sec": 3720,
        "notes": "first count"
      },
      {
        "phase": "closure",
        "start_sec": 3720,
        "end_sec": 4200,
        "notes": ""
      },
      {
        "phase": "turnover_room_reset",
        "start_sec": 4200,
        "end_sec": 4500,
        "notes": ""
      }
    ]
  }

  RULES:
    - Phases must cover the entire video (no gaps)
    - Phases must not overlap
    - Only valid transitions allowed (see config.yaml)

================================================================
IDLE LABEL FORMAT
================================================================

  File: {video_stem}_idle.json

  {
    "video": "sheba_general_001.mp4",
    "idle_periods": [
      {
        "start_sec": 600,
        "end_sec": 720,
        "type": "waiting_for_specimen"
      },
      {
        "start_sec": 2400,
        "end_sec": 2520,
        "type": "surgeon_break"
      }
    ]
  }

  IDLE TYPES:
    - waiting_for_specimen
    - surgeon_break
    - equipment_issue
    - room_turnover
    - patient_positioning
    - other

================================================================
LABELING GUIDELINES — WHAT IS IDLE vs ACTIVE
================================================================

  IDLE (label = 1):
    - Nobody working, room is quiet
    - Waiting for frozen section / pathology
    - Equipment malfunction, everyone standing around
    - Room turnover between cases

  ACTIVE (label = 0) — even if low motion:
    - Fine suturing (slow deliberate movement = still surgery!)
    - Delicate dissection
    - Surgeon studying anatomy before next cut
    - Waiting for pathology but team is prepping next step

  KEY RULE: If the surgeon is engaged, it's ACTIVE.
  Only label IDLE when there is genuinely no productive work.

================================================================
HOW TO RUN
================================================================

  PREPARE:
    python data_prep/prepare_data.py \
      --config config/config.yaml \
      --video-dir data/videos \
      --phase-dir data/phase_labels \
      --idle-dir data/idle_labels \
      --output-dir data

  TRAIN (phase recognition):
    python train/train_phase.py --config config/config.yaml --data-dir data

  TRAIN (idle detection):
    python train/train_idle.py --config config/config.yaml --data-dir data

  EVALUATE:
    python eval/evaluate.py --config config/config.yaml --data-dir data

================================================================
ACCEPTANCE CRITERIA
================================================================

  Phase Recognition:
    F1-score per phase:         >= 0.88
    Phase boundary accuracy:    ±30 seconds

  Idle Detection:
    Precision:                  >= 0.90
    Recall:                     >= 0.85
    Duration error:             < 10%
================================================================
