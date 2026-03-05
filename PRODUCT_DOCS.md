# TraciMedAlg — Product & Design Document

**Version:** 1.0.0 | **Date:** 2026-03-04 | **Repo:** https://github.com/asielad-hash/TraciMedAlg

---

## 1. Product Requirements (PRD)

### 1.1 Purpose
Fully automated surgical instrument detection and tracking pipeline using GroundingDINO (zero-shot detection) + SAM2 (mask propagation). Detects all medical tools in the first video frame via text prompt, generates segmentation masks, and propagates them through the entire video.

### 1.2 Target Users
- Research teams evaluating automated instrument detection
- Clinical engineers benchmarking detection accuracy
- Data scientists training downstream tracking models

### 1.3 Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Zero-Shot Detection | GroundingDINO detects instruments from text prompt ("medical tool.") — no training needed |
| 2 | Instance Segmentation | SAM2 generates per-object binary masks from detection bounding boxes |
| 3 | Video Propagation | SAM2 video predictor propagates masks across all frames with temporal consistency |
| 4 | Resolution Scaling | Configurable scale_factor (default 0.5) to trade speed for accuracy |
| 5 | Frame Range | Process subset of video (start_idx to end_idx) |
| 6 | Visualization | Optional debug overlays showing detections and masks |
| 7 | GPU Optimization | BFloat16 autocast + TF32 for Ampere+ GPUs |

### 1.4 User Workflow
```
Configure ToolTracking(source_video, text_prompt, thresholds)
  -> extract_frames()      Extract & scale video frames to JPEG directory
  -> load_models()         Load SAM2 + GroundingDINO onto GPU
  -> detect_objects()      Run GroundingDINO on frame 0
  -> generate_masks()      SAM2 image predictor creates per-object masks
  -> track_video()         SAM2 video predictor propagates masks → output MP4
  -> cleanup()             Free GPU memory
```

Or simply: `tracker.run(visualize=False)` for the full pipeline.

---

## 2. Software Design (SDD)

### 2.1 Tech Stack
| Layer | Technology |
|-------|-----------|
| Detection | GroundingDINO (HuggingFace: IDEA-Research/grounding-dino-base) |
| Segmentation | SAM2.1 Hiera Large (3.3 GB checkpoint) |
| Video I/O | OpenCV, Supervision (frame extraction + annotated video writing) |
| Deep Learning | PyTorch 2.7.0 + CUDA 12.6, BFloat16 + TF32 |
| Logging | Loguru (structured logging) |
| Environment | Conda (sam3 env), Windows 11 |

### 2.2 Architecture
```
ToolTracking (orchestrator class)
  |
  |-- extract_frames()
  |     └── supervision.get_video_frames_generator() → JPEG dir
  |
  |-- load_models()
  |     ├── build_sam2_video_predictor(config, checkpoint)
  |     ├── SAM2ImagePredictor(build_sam2(config, checkpoint))
  |     └── AutoProcessor + AutoModelForZeroShotObjectDetection (GroundingDINO)
  |
  |-- detect_objects()
  |     └── GroundingDINO inference on frame 0 → bounding boxes + labels
  |
  |-- generate_masks()
  |     └── SAM2 image predictor (multimask, best by IoU) → binary masks
  |
  |-- track_video()
  |     ├── init_state() from frame directory
  |     ├── add_new_mask() per detected object
  |     ├── propagate_in_video() → per-frame mask logits
  |     └── MaskAnnotator + VideoSink → output MP4
  |
  └── cleanup()
        └── torch.cuda.empty_cache()
```

### 2.3 File Structure
```
TraciMedAlg/
  tool_tracking.py             Main pipeline class (490 lines)
  track_grounded.ipynb         Jupyter notebook demo with sample results
```

### 2.4 Key Class: ToolTracking

```python
class ToolTracking:
    __init__(
        source_video: str,           # Input MP4 path
        text_prompt: str = "medical tool.",  # Detection prompt (lowercase + dot)
        scale_factor: float = 0.5,   # Resolution scale (0.5 = half res)
        start_idx: int = 0,          # First frame to process
        end_idx: int = 1000,         # Last frame to process
        box_threshold: float = 0.40, # GroundingDINO confidence threshold
        text_threshold: float = 0.50,# Text matching threshold
        checkpoint_path: str,        # SAM2 checkpoint (.pt)
        config_path: str,            # SAM2 config (.yaml)
        grounding_model_id: str,     # HuggingFace model ID
        output_suffix: str = "-result"
    )

    extract_frames() -> None
    load_models() -> None
    detect_objects() -> tuple[np.ndarray, list[str]]
    generate_masks() -> np.ndarray  # (N, H, W) binary
    track_video() -> Path           # Output MP4 path
    visualize_detections() -> None
    visualize_masks() -> None
    cleanup() -> None
    run(visualize=False) -> Path    # Full pipeline
```

### 2.5 Data Flow

**Detection Output (GroundingDINO):**
```python
input_boxes: np.ndarray   # Shape (N, 4) — [x1, y1, x2, y2]
labels: list[str]          # ["medical tool"] × N
```

**Mask Output (SAM2 Image):**
```python
best_masks: np.ndarray     # Shape (N, H, W) — binary masks
```

**Propagation Output (SAM2 Video):**
```python
frame_idx: int             # Current frame
object_ids: np.ndarray     # Tracked object IDs
mask_logits: Tensor        # Shape (N, 1, H, W), threshold > 0.0
```

### 2.6 Key Design Decisions
- **Text prompt format**: Must be lowercase ending with period ("medical tool.")
- **First-frame detection only**: All objects must be visible in frame 0
- **Multimask output + IoU selection**: SAM2 generates 3 masks per box, best selected by predicted IoU
- **scale_factor = 0.5**: Halves resolution for 4× speed improvement with acceptable accuracy loss
- **Frame directory approach**: SAM2 video predictor requires JPEG frames in numbered directory
- **No fill_hole_area issue**: Uses SAM2.1 (not SAM3), default parameters work correctly

### 2.7 Performance
From notebook (demo 7.mp4, 1000 frames, 16 objects, 960×540):
- Frame extraction: ~38 seconds
- Model loading: ~5 seconds
- Detection: < 1 second
- Mask generation: < 1 second
- Video propagation: ~35 minutes
- Output: MP4 with mask overlays

### 2.8 Prerequisites
- Windows 11 with NVIDIA GPU (CUDA 12.6 compatible)
- Miniconda / Anaconda
- SAM2.1 Hiera Large checkpoint (3.3 GB)
- SAM2 config: `sam2.1_hiera_l.yaml`

### 2.9 Installation
```bash
# 1. Create / activate conda environment
conda activate sam3

# 2. Install PyTorch with CUDA
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install SAM2
pip install sam2  # or: git clone + pip install -e .

# 4. Install additional dependencies
pip install supervision loguru tqdm transformers
```

### 2.10 Running
```bash
conda activate sam3
cd TraciMedAlg

# Option 1: Python script
python tool_tracking.py

# Option 2: Jupyter notebook (interactive)
jupyter notebook track_grounded.ipynb
```

**Programmatic usage:**
```python
from tool_tracking import ToolTracking

tracker = ToolTracking(
    source_video=r"path\to\video.mp4",
    text_prompt="medical tool.",
    scale_factor=0.5,
    start_idx=0,
    end_idx=1000,
    box_threshold=0.40,
    text_threshold=0.50
)
output_path = tracker.run(visualize=False)
tracker.cleanup()
# Output: video-result.mp4 in same directory
```

**Notes:**
- No web server — this is a batch processing pipeline
- Output video is saved alongside the source with `-result` suffix
- Temporary JPEG frames are extracted to a directory named after the video
- GPU memory is freed by calling `cleanup()` or using `run()` which calls it automatically
