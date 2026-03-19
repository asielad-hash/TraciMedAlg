"""
TrackiMed Training Framework — Shared Utilities
=================================================
Reusable code for all algorithm components.
Extracted from trackInstruments/ production codebase.

Modules:
  config           — YAML config loading, path resolution
  video_io         — Video frame extraction, metadata, I/O
  sam3_utils        — SAM3 model loading, session management
  mask_utils        — Mask rendering, identity swap correction
  annotation_io    — Keyframe/COCO annotation loading and conversion
  event_detection  — LOST/NEW/BACK event state machine
  cv_utils          — Homography, mask warping, color conversion
  eval_metrics     — IoU, HOTA, AUC, F1, WER, latency
  visualization    — Overlay rendering, matplotlib plots
"""
