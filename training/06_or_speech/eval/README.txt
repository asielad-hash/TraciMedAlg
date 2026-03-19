================================================================
06 — OR SPEECH: Planned Evaluation Metrics
STATUS: PLANNED — awaiting Phase 1 audio data
================================================================

EVALUATION METRICS:

  1. WORD ERROR RATE (WER)
     - Primary metric for overall transcription quality
     - Target: <= 10% on OR surgical vocabulary
     - Evaluated on: held-out test set of OR recordings
     - Breakdown: general speech WER vs OR-vocabulary-only WER

  2. COUNT CALL RECOGNITION RATE
     - Critical metric — count calls validate instrument tracking
     - Target: >= 95%
     - Test set: dedicated count call corpus
     - Measures: exact phrase match + semantic match

  3. INSTRUMENT NAME ACCURACY
     - How often instrument names are correctly transcribed
     - Target: >= 90%
     - Test on: all surgical vocabulary terms from config
     - Common confusions to watch: similar-sounding instruments

  4. STREAMING LATENCY
     - End-to-end time from speech to transcript output
     - Target: < 2 seconds
     - Measured on: live audio stream simulation

  5. NOISE ROBUSTNESS
     - WER at different SNR levels (5, 10, 15, 20 dB)
     - Must maintain WER <= 15% at SNR = 10 dB

SCRIPTS TO BE CREATED:
  evaluate_wer.py        — WER computation on test set
  evaluate_count_calls.py — Count call recognition accuracy
  evaluate_streaming.py  — Latency profiling
  evaluate_noise.py      — WER at different noise levels

GROUND TRUTH:
  - Manual transcriptions by clinical annotators
  - Inter-annotator agreement: Cohen's kappa >= 0.85
================================================================
