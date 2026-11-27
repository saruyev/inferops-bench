# Model-choice spike

Benchmarks the candidates for four paribus components on one Apple-silicon Mac and records
which to use:

| component | candidates |
|---|---|
| VAD | Silero VAD v6 (default threshold, 512-sample windows), WebRTC VAD aggressiveness 0-3 |
| audio tagger | EfficientAT `mn10_as` (CPU, MPS), AST `MIT/ast-finetuned-audioset-10-10-0.4593` (CPU, MPS) |
| language ID | SpeechBrain `lang-id-voxlingua107-ecapa`, Whisper large-v3-turbo `detect_language` |
| local ASR | Whisper large-v3-turbo (mlx-whisper), Parakeet TDT 0.6B v3 (parakeet-mlx), Qwen3-ASR 1.7B and 0.6B (mlx-audio, 8-bit) |

Results: [`results/summary.md`](results/summary.md) (tables, recommendations, licenses) and
`results/<component>__<candidate>.json` (config, environment, aggregate metrics, per-clip
outputs).

The spike is self-contained: its own `pyproject.toml` / `uv.lock` (Python 3.12, because the
torch/speechbrain/MLX stack is not ready for 3.14), nothing added to the paribus package.

## Rerun

```bash
cd spikes/model_choices
uv sync
uv run python prepare.py          # downloads ~1.1 GB of data, builds 264 clips + manifest
uv run python bench.py all        # each candidate in its own process; ~7.5 GB of weights
uv run python bench.py summary    # re-render results/summary.md only
```

One candidate: `uv run python bench.py run asr qwen3_asr_1.7b` (see `CANDIDATES` in
`bench.py`). A subset of components: `uv run python bench.py all --only vad lid`.

Everything downloaded (datasets, Hugging Face weights, the EfficientAT checkout at a pinned
commit, its release weights) lives under `~/.cache/paribus-spike` (override with
`PARIBUS_SPIKE_CACHE`). No audio or weights are committed.

## Test set

`prepare.py` builds 264 clips with a fixed seed:

- clean speech: 100 LibriSpeech test-clean (en), 50 FLEURS test (ru), 20 FLEURS test (he; for
  LID and unsupported-language behaviour), each 3-20 s;
- corrupted copies of 10 of those utterances (7 en, 3 ru), the same 10 in every condition:
  5-20 s lead/tail padding with digital zero or a -60 dBFS noise floor, a MUSAN music bed at
  SNR 0 and 10 dB, a MUSAN noise bed at SNR 0, 5 and 10 dB;
- non-speech clips of 5-30 s: 8 music-only, 8 noise-only, 4 digital silence, 4 noise floor.

`clips/manifest.jsonl` holds the ground truth per clip: condition, language, reference text,
speech segments and seconds (the original utterance counts as speech end to end, padded regions
as non-speech), music/noise presence, background source, SNR. `results/dataset.json` records the
composition, parameters and every dataset's URL and license.

MUSAN comes from a partial Hugging Face mirror (Akiki-AI `akiki-vad-silero623-v1-musan-*`,
original MUSAN audio with per-recording licenses) because the official MUSAN release is a single
11 GB tarball. Only the audio column is used.

## Notes on measurement

- RTF excludes model load and one warm-up call. VADs run on one CPU thread; MLX models run on
  the GPU; torch models on CPU unless the candidate name says `_mps`.
- Peak RSS is `ru_maxrss` of the candidate's own process. MLX keeps weights in Metal buffers that
  RSS only partly shows, so `mlx_peak_mb` (`mx.get_peak_memory()`) is reported next to it.
- WER uses jiwer after a simple normalization (NFKC, lowercase, strip punctuation, Russian
  ё -> е). No number normalization, so "twenty" vs "20" counts as an error for every model equally.
