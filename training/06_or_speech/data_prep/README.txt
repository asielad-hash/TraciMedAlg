================================================================
06 — OR SPEECH: Data Collection Requirements
STATUS: PLANNED — awaiting Phase 1 audio data
================================================================

WHAT DATA IS NEEDED:

  1. AUDIO EXTRACTION
     - Extract audio from anonymized OR videos (16kHz mono WAV)
     - Tool: ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 output.wav

  2. MANUAL TRANSCRIPTION
     - Clinical annotators transcribe speech segments
     - Focus on: count calls, instrument requests, medication calls
     - Format: JSON with start_sec, end_sec, text, speaker, category

  3. NOISE SAMPLES (for augmentation during training)
     - Suction device sounds (60+ seconds)
     - Monitor alarm beeps
     - Electrosurgery/cautery sounds
     - Ventilator cycling
     - Ambient OR chatter
     - Format: 16kHz mono WAV

  4. VOICE ACTIVITY DETECTION (VAD)
     - Silero VAD will segment speech regions
     - Reduces GPU load ~60% by skipping silence

WHEN TO START:
  Data collection begins when Phase 1 OR recordings are available.
  Script development begins when ≥20 transcribed procedures exist.

SCRIPTS TO BE CREATED:
  extract_audio.py       — ffmpeg wrapper to extract audio from videos
  run_vad.py             — Silero VAD segmentation
  prepare_transcripts.py — Organize + validate transcript JSONs
  prepare_splits.py      — Train/val/test split
================================================================
