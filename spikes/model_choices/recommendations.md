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
