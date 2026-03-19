================================================================
06 — OR SPEECH (Future Requirement)
Transcribe OR Speech for Medication, Instruments, Count Calls
================================================================

STATUS: PLANNED — not yet in active development.
Development begins when Phase 1 audio data is available.

================================================================
WHY THIS COMPONENT IS NEEDED
================================================================

  OR Speech is NOT a standalone feature. It provides input
  signals to ALL other engines:

  SIGNAL                    FEEDS INTO
  ─────────────────────────────────────────────────────────
  Count calls               Instrument Tracker (#01) — verify counts
  ("sponge count: ten")

  Instrument requests       Tray Monitor (#02) — predict TAKE_OUT
  ("give me the Kocher")

  Medication calls          Future Medication Safety Engine
  ("100mg cefazolin IV")

  Workflow cues              Surgical Workflow (#03) — phase hints
  ("let's do the count")

================================================================
EXPECTED DATA FOLDER STRUCTURE (when collection begins)
================================================================

data/
├── audio/                      ← Extracted audio (16kHz mono WAV)
│   ├── sheba_general_001.wav
│   └── ...
│
├── transcripts/                ← Human-verified transcripts
│   ├── sheba_general_001_transcript.json
│   └── ...
│
├── noise_samples/              ← OR ambient noise for training
│   ├── suction_device.wav      ← surgical suction sounds
│   ├── monitor_alarm.wav       ← patient monitor beeps
│   ├── electrosurgery.wav      ← cautery/bovie sounds
│   ├── ventilator.wav          ← ventilator cycling
│   └── ambient_chatter.wav     ← background OR conversation
│
├── train.json                  ← (auto-generated when ready)
├── val.json
├── test.json
├── count_calls.json            ← Focused count call corpus
└── surgical_vocabulary.json    ← OR term reference

================================================================
TRANSCRIPT FORMAT (when annotation begins)
================================================================

  File: {video_stem}_transcript.json

  [
    {
      "start_sec": 120.5,
      "end_sec": 123.0,
      "text": "sponge count: ten",
      "speaker": "nurse",
      "category": "count_call"
    },
    {
      "start_sec": 340.0,
      "end_sec": 342.5,
      "text": "give me the metzenbaum scissors",
      "speaker": "surgeon",
      "category": "instrument_request"
    },
    {
      "start_sec": 500.0,
      "end_sec": 502.0,
      "text": "100 mg cefazolin IV",
      "speaker": "anesthesiologist",
      "category": "medication_call"
    },
    {
      "start_sec": 3600.0,
      "end_sec": 3602.0,
      "text": "let's do the count",
      "speaker": "surgeon",
      "category": "workflow_cue"
    }
  ]

  CATEGORIES:
    - count_call           — any count-related utterance
    - instrument_request   — surgeon requesting an instrument
    - medication_call      — drug administration announcement
    - workflow_cue         — phase transition verbal signal
    - general              — other OR conversation

================================================================
NOISE SAMPLES REQUIREMENTS
================================================================

  Each noise file should be:
    - 16kHz mono WAV
    - At least 60 seconds long
    - Recorded in actual OR environment
    - Representative of typical noise levels

  These get mixed with clean speech at 5-20 dB SNR during training.

================================================================
KEY OR VOCABULARY THE MODEL MUST RECOGNIZE
================================================================

  Instruments:
    hemostatic, kocher, ray-tec, metzenbaum, babcock,
    weitlaner, gelpi, richardson, army-navy, mayo, kelly,
    allis, debakey, adson, castroviejo

  Materials:
    surgicel, gelfoam, cottonoid, lap sponge, raytec

  Drugs:
    epinephrine, lidocaine, heparin, marcaine, bupivacaine,
    cefazolin, vancomycin, fentanyl, propofol

  Count phrases:
    "count correct", "count verified", "all accounted for",
    "missing", "sponge count", "needle count", "sharp count"

================================================================
ACCEPTANCE CRITERIA (when trained)
================================================================

  Word Error Rate (WER):           <= 10% on OR vocabulary
  Count call recognition:          >= 95%
  Instrument name accuracy:        >= 90%
  Streaming latency:               < 2 seconds
================================================================
