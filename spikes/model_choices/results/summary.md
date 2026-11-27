# Model-choice spike: VAD, audio tagger, LID, local ASR

Machine: Apple M1 Pro, 32 GB, macOS-27.2.1-arm64-arm-64bit; Python 3.12.16. Package versions are in each results/*.json (`env`).

Test set: 264 clips, 51.0 min of 16 kHz mono audio, seed 20261127. Clips per condition: clean 170, pad_zero 10, pad_floor 10, music_snr0 10, music_snr10 10, noise_snr0 10, noise_snr5 10, noise_snr10 10, only_music 8, only_noise 8, only_zero 4, only_floor 4.

Ground truth: Original utterances count as speech end to end (internal pauses included); padded regions, music-only, noise-only and silence clips are non-speech. music_present / noise_present mark an added music or noise bed; the -60 dBFS noise floor is not counted as noise.

RTF = compute seconds / audio seconds, model load and one warm-up call excluded. Peak RSS is the process maximum (Python + framework + model); for MLX models the Metal allocator peak is listed separately because unified-memory buffers are not fully visible in RSS.

## Recommendations

Sample sizes are small: 10 source utterances per corruption, 24 non-speech clips, one
machine. A 1-2 point WER gap between candidates is noise; the large gaps below are not.

**VAD: Silero VAD v6.** It produced zero false speech on every music-only, noise-only and
silence clip and in every padded region, and still finds speech at 0 dB music or noise (16-17 %
"missed", about the same as clean, which is mostly inter-word pauses under our whole-utterance
ground truth). One CPU thread runs it at RTF 0.003, about 300x real time. WebRTC VAD at
aggressiveness 0-2 marks 82-99 % of music and noise as speech, so it cannot tell paribus where
the non-speech is, and that is the job here. At aggressiveness 3 it still flags 16-24 % of
non-speech and misses about 40 % of speech at 0 dB. WebRTC is fine only as a near-free energy
gate. Follow-up: fix Silero's segment-merging parameters (`min_silence_duration_ms`,
`speech_pad_ms`) when paribus defines its speech-ratio slice, so that pauses inside an
utterance do not count as non-speech.

**Audio tagger: AST (`MIT/ast-finetuned-audioset-10-10-0.4593`), on MPS.** Music AUC 1.00,
speech AUC 1.00, noise AUC 0.85, RTF 0.012 on MPS (0.032 on CPU), 600-850 MB RSS (MPS buffers are only partly in RSS). EfficientAT
mn10_as is lighter (RTF 0.018 on CPU, 508 MB) but scores music AUC 0.88 and noise AUC 0.70
here. One reason is that our 16 kHz input leaves the upper half of its 32 kHz mel range empty,
and paribus input will often be 8-16 kHz, so this shift is real and not a test artefact. It
also tags clean speech as "Clicking". On MPS it is no faster than on CPU (RTF 0.017 vs 0.018) and peaks at 4.5 GB RSS.
Three rules hold for both taggers:
- Both report "Music" (around 0.25) on digital silence, so gate tagger windows by energy or VAD
  before trusting the music score.
- Noise is the weak class. At a 0.5 threshold AST catches only 24 % of noise clips, so
  thresholds must be calibrated per class (or used as ranks), not fixed at 0.5.
- The music/noise/speech scores are max-pooled over AudioSet subtrees (mapping in
  `audioset_class_mapping.json`).

**Language ID: SpeechBrain VoxLingua107 ECAPA on at least 3 s of VAD speech, with Whisper as a
second opinion.** SpeechBrain costs 15-57 ms per call on CPU. Whisper-turbo `detect_language`
costs about 590 ms per call on the GPU, because it always encodes a 30 s window. Open-set,
SpeechBrain confuses Russian with Belarusian (Russian accuracy 62 % on full clips, 20 % on 1 s
crops), and it drops to 70-90 % on the corrupted en/ru clips. Restricted to the expected languages
(en/ru/he), it reaches 99.4 % at 3 s and 100 % on full clips. Whisper is the stronger open-set
classifier: 99.4 % on full clips, 100 % on every corrupted condition, though Hebrew on 1 s crops
only reaches 45 %. Proposed design:
- run SpeechBrain on every clip and keep its full posterior;
- resolve the label against a user-declared expected-language set when the user gives one;
- send low-margin or unexpected labels to Whisper.

When the pipeline under test is Whisper or Qwen3-ASR, their own detected language comes for
free and should be logged as well. Below 3 s neither model is reliable enough to slice on.

**Local ASR: Parakeet TDT 0.6B v3 (parakeet-mlx) as the default reference, Qwen3-ASR 1.7B as the
accuracy referee, Whisper large-v3-turbo as the system under test.**
- *Parakeet v3*: WER 2.5 % en and 6.4 % ru on clean speech. Padding does not hurt it (5.0 %
  zero-padded). It emitted 3 words on 1 of 24 non-speech clips. RTF 0.027, about 1.2 GB RSS
  (2.8 GB Metal peak). It gives token timestamps and token confidences; mean token confidence
  flags non-speech output with AUC 0.93. It does not report language and is limited to 25
  European languages.
- *Qwen3-ASR 1.7B (8-bit)*: the most accurate (2.3 % en, 4.5 % ru) and the most robust under a
  0 dB music bed (7.8 % vs Parakeet's 12.3 %), with zero output on non-speech. It is 2.4x
  slower (RTF 0.065) and heavier (2.8 GB RSS, 3.9 GB Metal peak). Word timestamps need the
  separate Qwen3-ForcedAligner. Token logprobs from mlx-audio are coarse (bf16 steps). Use it
  where Parakeet's own confidence is low, or to arbitrate fixes.
- *Qwen3-ASR 0.6B*: no advantage over Parakeet (7.8 % ru, 17.4 % at 0 dB music). Drop it.
- *Whisper large-v3-turbo*: good on clean speech (3.1 % en, 5.5 % ru) and the only candidate
  that handles Hebrew (35.5 %). But it wrote text on all 24 non-speech clips ("thank you" and
  repetition loops; 592 words without word timestamps, 66 with), and zero padding raised its
  WER from 5.5 % to 27-44 %. `no_speech_prob` came out 0.0 on all 379 segments in mlx-whisper,
  so that signal is unusable; `avg_logprob` (AUC 0.95) and mean word probability (AUC 0.997)
  do flag the hallucinations. It is 5-8x slower than Parakeet (RTF 0.15-0.21). These are
  exactly the failure modes paribus is meant to find, so keep Whisper as the main target
  pipeline in examples and tests, not as the reference.
- Unsupported languages: on Hebrew, Parakeet and Qwen3-ASR produce transliterated or
  wrong-script text (WER 81-122 %), and Qwen 0.6B labels all 20 clips "Arabic". paribus must
  gate WER and fix proposals on LID and each model's supported-language list.

## VAD

| candidate | RTF (1 thread) | speech-ratio err (speech clips) | missed, clean | missed, music 0 dB | missed, noise 0 dB | false speech, music-only | false speech, noise-only | false speech, silence | false s/min in zero pad | false s/min in floor pad |
|---|---|---|---|---|---|---|---|---|---|---|
| silero_v6 | 0.00333 | 0.167 | 18.2% | 16.7% | 16.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0.00 |
| webrtc_0 | 0.00014 | 0.072 | 8.9% | 0.1% | 1.5% | 98.6% | 86.2% | 3.4% | 0.58 | 10.36 |
| webrtc_1 | 0.00014 | 0.077 | 9.6% | 0.1% | 1.6% | 98.5% | 85.4% | 3.2% | 0.61 | 9.94 |
| webrtc_2 | 0.00014 | 0.100 | 12.2% | 0.6% | 3.1% | 98.0% | 82.1% | 2.9% | 0.33 | 8.26 |
| webrtc_3 | 0.00013 | 0.271 | 24.9% | 42.5% | 38.8% | 15.6% | 24.4% | 0.1% | 0.01 | 0.77 |

Speech-ratio error = |predicted speech fraction - true speech fraction| per clip. Ground truth counts the whole utterance as speech, so pauses inside an utterance count as 'missed' for a VAD that (correctly) closes segments on pauses; compare the 'missed' columns between candidates rather than reading them as absolute.

## Audio tagger

| candidate | device | music AUC | noise AUC | speech AUC | music TPR/FPR @0.5 | noise TPR/FPR @0.5 | RTF | load s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| ast_cpu | cpu | 1.000 | 0.852 | 1.000 | 0.71 / 0.00 | 0.24 / 0.02 | 0.0323 | 8.3 | 852 |
| ast_mps | mps | 1.000 | 0.852 | 1.000 | 0.71 / 0.00 | 0.24 / 0.02 | 0.0122 | 5.2 | 600 |
| efficientat_mn10_cpu | cpu | 0.881 | 0.701 | 0.922 | 0.96 / 0.14 | 0.42 / 0.16 | 0.0177 | 2.7 | 508 |
| efficientat_mn10_mps | mps | 0.881 | 0.701 | 0.922 | 0.96 / 0.14 | 0.42 / 0.16 | 0.0169 | 3.9 | 4554 |

Class mapping (AudioSet ontology subtrees, 527 classes) is in `results/audioset_class_mapping.json`. Positives: music bed (SNR 0/10 dB) and music-only clips for music; noise bed (SNR 0/5/10 dB) and noise-only clips for noise. The -60 dBFS floor and digital silence count as negatives.

## Language ID

| candidate | acc 1 s (en/ru/he) | acc 3 s (en/ru/he) | acc full (en/ru/he) | acc restricted to {en,ru,he}, 1s/3s/full | ms/call 1s/3s/full | load s | peak RSS MB |
|---|---|---|---|---|---|---|---|
| speechbrain_ecapa | 59.0% / 20.0% / 70.0% | 95.0% / 42.0% / 90.0% | 98.0% / 62.0% / 100.0% | 94.1% / 99.4% / 100.0% | 15 / 22 / 57 | 3.8 | 1530 |
| whisper_turbo | 100.0% / 86.0% / 45.0% | 100.0% / 98.0% / 85.0% | 100.0% / 100.0% / 95.0% | 91.8% / 98.2% / 100.0% | 592 / 591 / 608 | 2.8 | 1847 |

Full-clip accuracy on corrupted copies (en+ru, 10 clips each):

| candidate | music_snr0 | music_snr10 | noise_snr0 | noise_snr10 | noise_snr5 | pad_floor | pad_zero |
|---|---|---|---|---|---|---|---|
| speechbrain_ecapa | 80.0% | 90.0% | 80.0% | 80.0% | 70.0% | 90.0% | 80.0% |
| whisper_turbo | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Crops are centered in the clean utterance. Accuracy columns are open-set top-1 (107 languages for SpeechBrain, 100 for Whisper); the restricted column takes the argmax over en/ru/he only, i.e. what a pipeline that knows its expected languages would see. Whisper always encodes a 30 s window, so its cost per call does not shrink with the crop.

## Local ASR

| candidate | runtime | weights | WER clean en | WER clean ru | WER clean he* | non-speech words / clips with output | RTF | load s | peak RSS MB | MLX peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| parakeet_v3 | parakeet-mlx | bfloat16 | 2.5% | 6.4% | 110.0% | 3 / 1 of 24 | 0.0271 | 1.9 | 1165 | 2752 |
| qwen3_asr_0.6b | mlx-audio | 8-bit quantized (mlx-community) | 2.5% | 7.8% | 122.5% | 0 / 0 of 24 | 0.0312 | 4.4 | 1443 | 2368 |
| qwen3_asr_1.7b | mlx-audio | 8-bit quantized (mlx-community) | 2.3% | 4.5% | 80.8% | 0 / 0 of 24 | 0.0649 | 11.5 | 2839 | 3857 |
| whisper_turbo | mlx-whisper | float16 | 3.1% | 5.5% | 35.5% | 66 / 24 of 24 | 0.2091 | 2.7 | 1964 | 2381 |
| whisper_turbo_nowords | mlx-whisper | float16 | 3.1% | 5.5% | 35.5% | 592 / 24 of 24 | 0.1461 | 4.5 | 1915 | 2381 |

\* Hebrew is outside the supported language list of Parakeet v3 and Qwen3-ASR; it is there to show unsupported-language behaviour.

WER on the 10 corruption sources (7 en + 3 ru), same utterances in every column:

| candidate | clean | pad_zero | pad_floor | music_snr10 | music_snr0 | noise_snr10 | noise_snr5 | noise_snr0 |
|---|---|---|---|---|---|---|---|---|
| parakeet_v3 | 5.9% | 5.0% | 6.8% | 6.8% | 12.3% | 5.9% | 6.8% | 10.0% |
| qwen3_asr_0.6b | 5.9% | 7.8% | 6.4% | 6.8% | 17.4% | 6.8% | 10.5% | 12.3% |
| qwen3_asr_1.7b | 4.6% | 4.1% | 6.4% | 5.0% | 7.8% | 6.8% | 8.2% | 6.8% |
| whisper_turbo | 5.5% | 26.9% | 13.2% | 7.3% | 8.7% | 6.4% | 6.8% | 10.0% |
| whisper_turbo_nowords | 5.5% | 44.3% | 16.4% | 7.3% | 8.7% | 6.4% | 6.8% | 10.0% |

Signals each runtime exposes:

| candidate | word timestamps | no_speech_prob | avg_logprob | token confidences | language |
|---|---|---|---|---|---|
| parakeet_v3 | token/word timestamps | False | False | True | False |
| qwen3_asr_0.6b | no (separate Qwen3-ForcedAligner model) | no ('language None' tag on no speech) | computable from stream_generate logprobs | token logprobs via stream_generate | detected language name in output |
| qwen3_asr_1.7b | no (separate Qwen3-ForcedAligner model) | no ('language None' tag on no speech) | computable from stream_generate logprobs | token logprobs via stream_generate | detected language name in output |
| whisper_turbo | True | True | True | word probability | detected language + probabilities |
| whisper_turbo_nowords | False | True | True | word probability | detected language + probabilities |

How well each exposed signal flags output that came from a non-speech clip (AUC over clips with any output; 1.0 = a threshold on the signal would drop every hallucination and keep every real transcript):

| candidate | signal | AUC | non-speech / all clips with output |
|---|---|---|---|
| parakeet_v3 | -token_conf_mean | 0.925 | 1 / 241 |
| whisper_turbo | max no_speech_prob | 0.500 | 24 / 264 |
| whisper_turbo | -mean avg_logprob | 0.952 | 24 / 264 |
| whisper_turbo | -word_prob_mean | 0.997 | 24 / 264 |
| whisper_turbo_nowords | max no_speech_prob | 0.500 | 24 / 264 |
| whisper_turbo_nowords | -mean avg_logprob | 0.949 | 24 / 264 |

Hallucination examples on non-speech clips (normalized text, first 80 chars):

- parakeet_v3 on `only_music`: "i don't know"
- whisper_turbo on `only_floor`: "thank you"
- whisper_turbo on `only_floor`: "thank you"
- whisper_turbo on `only_music`: "launch comme là dans"
- whisper_turbo on `only_music`: "thank you"
- whisper_turbo on `only_noise`: "countdown 을"
- whisper_turbo on `only_noise`: "thank you"
- whisper_turbo on `only_zero`: "thank you"
- whisper_turbo on `only_zero`: "thank you"
- whisper_turbo_nowords on `only_floor`: "thank you"
- whisper_turbo_nowords on `only_floor`: "thank you"
- whisper_turbo_nowords on `only_music`: "i have a no durl i have a no durl i have a no durl i have a no durl i have a no "
- whisper_turbo_nowords on `only_music`: "thank you"
- whisper_turbo_nowords on `only_noise`: "i have aose cow a p m i have a s a b a t a t a i have a t a i have t a i have t "
- whisper_turbo_nowords on `only_noise`: "thank you"
- whisper_turbo_nowords on `only_zero`: "thank you"
- whisper_turbo_nowords on `only_zero`: "thank you"

## Licenses

| component | source | code license | weights license |
|---|---|---|---|
| Silero VAD v6 | silero-vad (PyPI) / snakers4/silero-vad | MIT | MIT (bundled weights) |
| WebRTC VAD | webrtcvad-wheels (PyPI) | MIT (wrapper); WebRTC code BSD-3-Clause | n/a (no learned weights) |
| EfficientAT mn10_as | github.com/fschmid56/EfficientAT | MIT | MIT (GitHub release v0.0.1, same repository); trained on AudioSet |
| AST AudioSet | MIT/ast-finetuned-audioset-10-10-0.4593 | transformers: Apache-2.0 | BSD-3-Clause (model card) |
| SpeechBrain VoxLingua107 ECAPA | speechbrain/lang-id-voxlingua107-ecapa | speechbrain: Apache-2.0 | Apache-2.0 (model card) |
| Whisper large-v3-turbo | openai/whisper-large-v3-turbo -> mlx-community port | mlx-whisper: MIT | MIT (model card) |
| Parakeet TDT 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 -> mlx-community port | parakeet-mlx: Apache-2.0 | CC-BY-4.0 (model card) |
| Qwen3-ASR 1.7B / 0.6B | Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B -> mlx-community 8-bit | mlx-audio: MIT | Apache-2.0 (model card) |

Datasets:

| dataset | url used | license (as stated) | note |
|---|---|---|---|
| librispeech_test_clean | https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/clean/test/0000.parquet | CC-BY-4.0 | openslr.org mirror ran at ~80 KB/s, so the identical HF parquet was used. |
| fleurs_ru_test | https://huggingface.co/datasets/google/fleurs/resolve/main/data/ru_ru/audio/test.tar.gz | CC-BY-4.0 |  |
| fleurs_he_test | https://huggingface.co/datasets/google/fleurs/resolve/main/data/he_il/audio/test.tar.gz | CC-BY-4.0 |  |
| musan_noise | https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-noise/resolve/main/data/musan_noise/train-00000.parquet | CC-BY-4.0 (per-recording audio_license field; MUSAN noise subset) | Partial MUSAN mirror (104 of 930 noise recordings, original audio, 16 kHz). Full MUSAN is an 11 GB tarball; only the audio column is used. |
| musan_music_cc_by_3 | https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-music-cc-by-3.0/resolve/main/data/musan_music/train-00000.parquet | CC-BY-3.0 (per-recording; MUSAN music, documented instrumental) |  |
| musan_music_cc_by_sa_3 | https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-music-cc-by-sa-3.0/resolve/main/data/musan_music/train-00000.parquet | CC-BY-SA-3.0 (per-recording; MUSAN music, documented instrumental) |  |
| audioset_ontology | https://raw.githubusercontent.com/audioset/ontology/master/ontology.json | CC-BY-SA-4.0 | Used only to build the music / noise / speech class mapping. |

## Failures and skips

- No candidate run failed.

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
