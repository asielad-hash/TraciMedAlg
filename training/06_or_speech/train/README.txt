================================================================
06 — OR SPEECH: Planned Training Approach
STATUS: PLANNED — awaiting Phase 1 audio data
================================================================

TRAINING APPROACH:

  1. BASE MODEL
     - Whisper large-v3 (OpenAI, open weights, 300M+ params)
     - Pre-trained on massive multilingual speech data
     - Already strong at English medical vocabulary

  2. FINE-TUNING STRATEGY
     - Freeze encoder (general speech features are already excellent)
     - Fine-tune decoder on OR-specific vocabulary
     - Focus training on:
       * Surgical instrument names (metzenbaum, kocher, ray-tec...)
       * Count call phrases ("sponge count: ten", "needle count correct")
       * Drug names (epinephrine, lidocaine, cefazolin...)
       * Workflow cues ("let's do the count", "knife to skin")

  3. OR NOISE AUGMENTATION
     - Mix clean speech with OR noise at 5-20 dB SNR
     - Noise sources: suction, alarms, cautery, ventilator
     - Makes model robust to real OR acoustic conditions

  4. STREAMING INFERENCE
     - Whisper-Streaming (chunked mode) for real-time
     - Silero VAD gate eliminates silent segments
     - Target: <2 second latency

SCRIPTS TO BE CREATED:
  train_whisper_or.py    — HuggingFace Trainer-based fine-tuning
  train_streaming.py     — Streaming inference optimization

DEPENDENCIES:
  pip install transformers datasets torchaudio
  pip install silero-vad

HOW SPEECH FEEDS OTHER ENGINES:
  Count calls        → Instrument Tracker (#01) count verification
  Instrument requests → Tray Monitor (#02) predict TAKE_OUT events
  Workflow cues       → Surgical Workflow (#03) phase transition hints
  Medication calls    → Future Medication Safety Engine
================================================================
