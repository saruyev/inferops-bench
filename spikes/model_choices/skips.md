Deviations from the plan, and why:

- **Qwen3-ASR ran as 8-bit MLX quantizations** (`mlx-community/Qwen3-ASR-{1.7B,0.6B}-8bit`).
  The bf16 1.7B port alone is 4.1 GB and would have pushed the total download past the
  ~10 GB budget (actual total: ~8.7 GB of data and weights). bf16 accuracy may be slightly
  higher.
- **No torch fallback was needed.** All four ASR models ran on MLX: mlx-whisper 0.4.3,
  parakeet-mlx 0.7.0, mlx-audio 0.6.8.
- **LibriSpeech came from the Hugging Face parquet mirror** (`openslr/librispeech_asr`), the
  same files as openslr.org, because the openslr.org download ran at ~80 KB/s.
- **MUSAN came from a partial mirror** (Akiki-AI `akiki-vad-silero623-v1-musan-*`: 96 usable
  noise recordings and 48 instrumental music recordings, original audio with per-recording
  licenses). Only the audio was used, not the mirror's Silero-generated labels or code. The
  noise subset is not hand-checked for incidental speech.
- **Whisper `no_speech_prob` was 0.0 on every segment** (large-v3-turbo in mlx-whisper), so its
  separability AUC of 0.5 measures a dead signal, not a weak one. Not investigated further.
- **EfficientAT on MPS** gains nothing over CPU (RTF 0.017 vs 0.018; 0.019 in a first run) and
  peaks at 3.3-4.5 GB RSS, likely from per-shape MPS graphs for variable-length input. It is
  reported, but CPU is its sensible path.
- **Hebrew WER** for Parakeet and Qwen3-ASR only shows unsupported-language behaviour; it is
  not a quality number.
- Whisper LID pads 1 s and 3 s crops to its fixed 30 s window, so its per-call cost is flat.
- RTF for the MLX models was measured with the GPU otherwise idle; the two model-download jobs
  had finished before the timed ASR runs.
