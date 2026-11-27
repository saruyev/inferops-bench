"""Benchmark VAD, audio tagger, LID and local ASR candidates on the spike test set.

Usage:
  uv run python bench.py all                     # every candidate, each in its own process
  uv run python bench.py run asr whisper_turbo   # one candidate
  uv run python bench.py summary                 # rebuild results/summary.md from results/*.json

Run prepare.py first. Each run writes results/<component>__<candidate>.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from common import CACHE, CLIPS, RAW, RESULTS_DIR, SR, load_manifest, normalize_text, write_json

EFFICIENTAT_REPO = "https://github.com/fschmid56/EfficientAT"
EFFICIENTAT_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_DIR = CACHE / "src" / "EfficientAT"

CANDIDATES: dict[str, list[str]] = {
    "vad": ["silero_v6", "webrtc_0", "webrtc_1", "webrtc_2", "webrtc_3"],
    "tagger": ["efficientat_mn10_cpu", "efficientat_mn10_mps", "ast_cpu", "ast_mps"],
    "lid": ["speechbrain_ecapa", "whisper_turbo"],
    "asr": [
        "whisper_turbo",
        "whisper_turbo_nowords",
        "parakeet_v3",
        "qwen3_asr_1.7b",
        "qwen3_asr_0.6b",
    ],
}

ASR_REPOS = {
    "whisper_turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper_turbo_nowords": "mlx-community/whisper-large-v3-turbo",
    "parakeet_v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "qwen3_asr_1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "qwen3_asr_0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
}
ASR_UPSTREAM = {
    "whisper_turbo": "openai/whisper-large-v3-turbo",
    "whisper_turbo_nowords": "openai/whisper-large-v3-turbo",
    "parakeet_v3": "nvidia/parakeet-tdt-0.6b-v3",
    "qwen3_asr_1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B",
}
AST_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"
LID_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
QWEN_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------- utilities


def load_audio(entry: dict[str, Any]) -> np.ndarray:
    audio, sr = sf.read(CLIPS / entry["path"], dtype="float32")
    assert sr == SR
    return audio


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KiB on Linux


def mlx_peak_mb() -> float | None:
    try:
        import mlx.core as mx

        return mx.get_peak_memory() / 2**20
    except Exception:
        return None


@contextlib.contextmanager
def timer(acc: dict[str, float], key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    with contextlib.suppress(Exception):
        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        info["mem_gb"] = (
            int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                ).stdout
            )
            / 2**30
        )
    from importlib.metadata import PackageNotFoundError, version

    pkgs = {}
    for p in (
        "torch",
        "torchaudio",
        "silero-vad",
        "webrtcvad-wheels",
        "speechbrain",
        "transformers",
        "mlx",
        "mlx-audio",
        "mlx-whisper",
        "parakeet-mlx",
        "jiwer",
    ):
        with contextlib.suppress(PackageNotFoundError):
            pkgs[p] = version(p)
    info["packages"] = pkgs
    return info


def run_over_clips(
    entries: list[dict[str, Any]], fn: Callable[[dict[str, Any], np.ndarray], dict], label: str
) -> tuple[list[dict[str, Any]], float, float]:
    """Apply ``fn`` to every clip; returns rows, total compute seconds, total audio seconds."""
    rows, compute, audio_s = [], 0.0, 0.0
    for i, e in enumerate(entries):
        audio = load_audio(e)
        t0 = time.perf_counter()
        out = fn(e, audio)
        dt = time.perf_counter() - t0
        compute += dt
        audio_s += e["duration_s"]
        rows.append({"clip_id": e["clip_id"], "condition": e["condition"], "time_s": dt, **out})
        if (i + 1) % 25 == 0:
            print(f"[{label}] {i + 1}/{len(entries)} rtf={compute / audio_s:.4f}", flush=True)
    return rows, compute, audio_s


def _mask(segments: list[list[float]], n: int, step: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in segments:
        m[round(s / step) : round(e / step)] = True
    return m


# ---------------------------------------------------------------------------- VAD


def bench_vad(candidate: str) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    entries = load_manifest()
    t0 = time.perf_counter()
    if candidate == "silero_v6":
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()  # TorchScript, 512-sample windows at 16 kHz

        def segments(audio: np.ndarray) -> list[list[float]]:
            ts = get_speech_timestamps(
                torch.from_numpy(audio), model, sampling_rate=SR, return_seconds=True
            )  # default threshold 0.5
            return [[float(t["start"]), float(t["end"])] for t in ts]

        config = {
            "package": "silero-vad",
            "threshold": 0.5,
            "window_samples": 512,
            "other": "get_speech_timestamps defaults (min_speech 250 ms, "
            "min_silence 100 ms, speech_pad 30 ms)",
            "threads": 1,
        }
    else:
        import webrtcvad

        mode = int(candidate.rsplit("_", 1)[1])
        vad = webrtcvad.Vad(mode)
        frame = int(0.03 * SR)

        def segments(audio: np.ndarray) -> list[list[float]]:
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            segs = []
            for k in range(len(pcm) // frame):
                if vad.is_speech(pcm[k * frame : (k + 1) * frame].tobytes(), SR):
                    segs.append([k * 0.03, (k + 1) * 0.03])
            return segs

        config = {
            "package": "webrtcvad-wheels",
            "aggressiveness": mode,
            "frame_ms": 30,
            "other": "raw per-frame decisions, no smoothing",
        }
    load_s = time.perf_counter() - t0
    step = 0.01

    def per_clip(e: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
        pred = segments(audio)
        n = int(np.ceil(e["duration_s"] / step))
        pm, tm = _mask(pred, n, step), _mask(e["speech_segments"], n, step)
        return {
            "pred_speech_s": float(pm.sum() * step),
            "true_speech_s": e["speech_s"],
            "duration_s": e["duration_s"],
            "ratio_error": float(abs(pm.mean() - tm.mean())),
            "false_speech_s": float((pm & ~tm).sum() * step),
            "missed_speech_s": float((~pm & tm).sum() * step),
            "n_segments": len(pred),
        }

    rows, compute, audio_s = run_over_clips(entries, per_clip, f"vad/{candidate}")
    by_cond: dict[str, dict[str, float]] = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        dur = sum(r["duration_s"] for r in rs)
        true = sum(r["true_speech_s"] for r in rs)
        by_cond[cond] = {
            "n": len(rs),
            "mean_ratio_error": float(np.mean([r["ratio_error"] for r in rs])),
            "false_speech_s_per_min": 60 * sum(r["false_speech_s"] for r in rs) / dur,
            "missed_speech_frac": (sum(r["missed_speech_s"] for r in rs) / true) if true else None,
        }

    def ns(conds: tuple[str, ...]) -> dict[str, float]:
        rs = [r for r in rows if r["condition"] in conds]
        dur = sum(r["duration_s"] for r in rs)
        fs = sum(r["false_speech_s"] for r in rs)
        return {
            "false_speech_s": fs,
            "audio_s": dur,
            "false_speech_frac": fs / dur,
            "clips_with_any_speech": sum(r["pred_speech_s"] > 0 for r in rs),
            "n": len(rs),
        }

    speech_rows = [r for r in rows if r["true_speech_s"] > 0]
    metrics = {
        "mean_ratio_error_all": float(np.mean([r["ratio_error"] for r in rows])),
        "mean_ratio_error_speech_clips": float(np.mean([r["ratio_error"] for r in speech_rows])),
        "missed_speech_frac_clean": by_cond["clean"]["missed_speech_frac"],
        "nonspeech_music": ns(("only_music",)),
        "nonspeech_noise": ns(("only_noise",)),
        "nonspeech_silence": ns(("only_zero", "only_floor")),
        "padding_false_speech_s_per_min": {
            c: by_cond[c]["false_speech_s_per_min"] for c in ("pad_zero", "pad_floor")
        },
        "by_condition": by_cond,
    }
    return {
        "config": config,
        "load_s": load_s,
        "compute_s": compute,
        "audio_s": audio_s,
        "rtf": compute / audio_s,
        "metrics": metrics,
        "per_clip": rows,
    }


# ---------------------------------------------------------------------------- tagger


def audioset_mapping() -> dict[str, Any]:
    """Map the 527 AudioSet classes to music / speech / noise via the AudioSet ontology."""
    import csv

    onto = {n["id"]: n for n in json.loads((RAW / "audioset" / "ontology.json").read_text())}
    by_name = {n["name"]: n["id"] for n in onto.values()}

    def subtree(name: str) -> set[str]:
        out, stack = set(), [by_name[name]]
        while stack:
            mid = stack.pop()
            if mid not in out:
                out.add(mid)
                stack.extend(onto[mid].get("child_ids", []))
        return out

    with (EFFICIENTAT_DIR / "metadata" / "class_labels_indices.csv").open() as fh:
        labels = [(int(r["index"]), r["mid"], r["display_name"]) for r in csv.DictReader(fh)]
    music, speech = subtree("Music"), subtree("Speech")
    excluded = (
        subtree("Human sounds")
        | subtree("Acoustic environment")
        | subtree("Sound reproduction")
        | subtree("Silence")
        | music
    )
    groups: dict[str, list[int]] = {"music": [], "speech": [], "noise": []}
    for idx, mid, _ in labels:
        if mid in music:
            groups["music"].append(idx)
        if mid in speech:
            groups["speech"].append(idx)
        if mid not in excluded:
            groups["noise"].append(idx)
    return {
        "rule": {
            "music": "max over the 'Music' subtree (instruments, genres, singing, moods)",
            "speech": "max over the 'Speech' subtree",
            "noise": "max over every class outside 'Music', 'Human sounds', "
            "'Acoustic environment', 'Sound reproduction' and 'Silence'",
            "clip_score": "max over non-overlapping 10 s windows",
        },
        "indices": groups,
        "names": {g: [labels[i][2] for i in idx] for g, idx in groups.items()},
        "labels": [name for _, _, name in labels],
    }


def bench_tagger(candidate: str) -> dict[str, Any]:
    import soxr
    import torch
    from sklearn.metrics import roc_auc_score

    device = "mps" if candidate.endswith("_mps") else "cpu"
    entries = load_manifest()
    mapping = audioset_mapping()
    idx = {g: np.array(v) for g, v in mapping["indices"].items()}
    window = 10.0
    t0 = time.perf_counter()
    if candidate.startswith("efficientat"):
        sys.path.insert(0, str(EFFICIENTAT_DIR))
        cwd = os.getcwd()
        os.chdir(EFFICIENTAT_DIR)  # helpers.utils reads metadata/ relative to cwd
        try:
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT

            model = get_model(width_mult=1.0, pretrained_name="mn10_as", head_type="mlp")
        finally:
            os.chdir(cwd)
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
        model.to(device).eval()
        mel.to(device).eval()

        def probs(chunk: np.ndarray) -> np.ndarray:
            wav = torch.from_numpy(soxr.resample(chunk, SR, 32000)[None, :]).to(device)
            with torch.no_grad():
                logits, _ = model(mel(wav).unsqueeze(0))
            return torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

        config = {
            "weights": "mn10_as_mAP_471.pt (GitHub release v0.0.1)",
            "code": f"{EFFICIENTAT_REPO}@{EFFICIENTAT_COMMIT}",
            "input": "16 kHz clip resampled to 32 kHz (no content above 8 kHz)",
        }
    else:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        fe = ASTFeatureExtractor.from_pretrained(AST_REPO)
        model = ASTForAudioClassification.from_pretrained(AST_REPO).to(device).eval()
        names = [model.config.id2label[i] for i in range(model.config.num_labels)]
        if names != mapping["labels"]:
            mismatch = sum(a != b for a, b in zip(names, mapping["labels"], strict=True))
            print(f"warning: {mismatch} AST label names differ from the AudioSet csv")

        def probs(chunk: np.ndarray) -> np.ndarray:
            feats = fe(chunk, sampling_rate=SR, return_tensors="pt")["input_values"].to(device)
            with torch.no_grad():
                logits = model(input_values=feats).logits
            return torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

        config = {"weights": AST_REPO, "input": "16 kHz, 10.24 s max per window"}
    config["device"] = device
    load_s = time.perf_counter() - t0
    probs(np.zeros(SR, np.float32))  # warm-up (kernel compilation on MPS)

    def per_clip(e: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
        n = int(window * SR)
        chunks = [audio[i : i + n] for i in range(0, len(audio), n)]
        if len(chunks) > 1 and len(chunks[-1]) < SR:
            chunks = chunks[:-1]
        p = np.max(np.stack([probs(c) for c in chunks]), axis=0)
        top = np.argsort(p)[::-1][:3]
        return {
            "music": float(p[idx["music"]].max()),
            "speech": float(p[idx["speech"]].max()),
            "noise": float(p[idx["noise"]].max()),
            "top3": [[mapping["labels"][i], round(float(p[i]), 3)] for i in top],
        }

    rows, compute, audio_s = run_over_clips(entries, per_clip, f"tagger/{candidate}")
    truth = {e["clip_id"]: e for e in entries}

    def auc_block(target: str, score: str) -> dict[str, Any]:
        y = np.array([bool(truth[r["clip_id"]][target]) for r in rows])
        s = np.array([r[score] for r in rows])
        return {
            "auc": float(roc_auc_score(y, s)),
            "n_pos": int(y.sum()),
            "n_neg": int((~y).sum()),
            "tpr_at_0.5": float((s[y] >= 0.5).mean()),
            "fpr_at_0.5": float((s[~y] >= 0.5).mean()),
        }

    by_cond = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        by_cond[cond] = {
            k: float(np.mean([r[k] for r in rs])) for k in ("music", "noise", "speech")
        }
    metrics = {
        "music_present": auc_block("music_present", "music"),
        "noise_present": auc_block("noise_present", "noise"),
        "speech_present": auc_block("has_speech", "speech"),
        "mean_scores_by_condition": by_cond,
        "peak_rss_mb": peak_rss_mb(),
    }
    write_json(
        RESULTS_DIR / "audioset_class_mapping.json",
        {k: v for k, v in mapping.items() if k != "labels"},
    )
    return {
        "config": config,
        "load_s": load_s,
        "compute_s": compute,
        "audio_s": audio_s,
        "rtf": compute / audio_s,
        "metrics": metrics,
        "per_clip": rows,
    }


# ---------------------------------------------------------------------------- LID

CROPS = {"1s": 1.0, "3s": 3.0, "full": None}
SB_LABELS = {"en": "en: English", "ru": "ru: Russian", "he": "iw: Hebrew"}


def _crop(audio: np.ndarray, seconds: float | None) -> np.ndarray:
    if seconds is None or len(audio) <= seconds * SR:
        return audio
    n = int(seconds * SR)
    start = (len(audio) - n) // 2  # centered crop
    return audio[start : start + n]


def bench_lid(candidate: str) -> dict[str, Any]:
    entries = [e for e in load_manifest() if e["has_speech"]]
    t0 = time.perf_counter()
    if candidate == "speechbrain_ecapa":
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        clf = EncoderClassifier.from_hparams(
            source=LID_REPO,
            savedir=str(CACHE / "speechbrain" / "lid"),
            local_strategy=LocalStrategy.COPY,
        )

        enc = clf.hparams.label_encoder
        subset = {k: enc.lab2ind[n] for k, n in SB_LABELS.items()}

        def detect(audio: np.ndarray) -> tuple[str, float, str]:
            with torch.no_grad():
                out_prob, score, _, lab = clf.classify_batch(torch.from_numpy(audio)[None])
            code = lab[0].split(":")[0].strip()
            restricted = max(subset, key=lambda k: float(out_prob[0, subset[k]]))
            return ("he" if code == "iw" else code), float(score.exp()[0]), restricted

        config = {"weights": LID_REPO, "device": "cpu", "labels": "VoxLingua107 (Hebrew = 'iw')"}
    else:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(ASR_REPOS["whisper_turbo"], mx.float16)

        def detect(audio: np.ndarray) -> tuple[str, float, str]:
            mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
            seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
            _, probs = model.detect_language(seg)
            lang = max(probs, key=probs.get)
            return lang, float(probs[lang]), max(SB_LABELS, key=lambda k: probs.get(k, 0.0))

        config = {
            "weights": ASR_REPOS["whisper_turbo"],
            "device": "mlx (GPU)",
            "method": "model.detect_language on the first 30 s (as in transcribe)",
        }
    load_s = time.perf_counter() - t0
    detect(np.zeros(SR, np.float32))

    def per_clip(e: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
        out = {}
        crops = CROPS if e["group"] == "clean" else {"full": None}
        for name, sec in crops.items():
            t = time.perf_counter()
            lang, conf, restricted = detect(_crop(audio, sec))
            out[name] = {
                "pred": lang,
                "conf": conf,
                "pred_en_ru_he": restricted,
                "time_s": time.perf_counter() - t,
            }
        out["language"] = e["language"]
        return out

    rows, compute, audio_s = run_over_clips(entries, per_clip, f"lid/{candidate}")
    acc: dict[str, Any] = {}
    for crop in CROPS:
        for lang in ("en", "ru", "he"):
            rs = [r for r in rows if r["condition"] == "clean" and r["language"] == lang]
            acc.setdefault(crop, {})[lang] = float(np.mean([r[crop]["pred"] == lang for r in rs]))
        rs = [r for r in rows if r["condition"] == "clean"]
        acc[crop]["all"] = float(np.mean([r[crop]["pred"] == r["language"] for r in rs]))
        acc[crop]["all_restricted_en_ru_he"] = float(
            np.mean([r[crop]["pred_en_ru_he"] == r["language"] for r in rs])
        )
        for lang in ("en", "ru", "he"):
            rl = [r for r in rs if r["language"] == lang]
            acc[crop][f"{lang}_restricted"] = float(
                np.mean([r[crop]["pred_en_ru_he"] == lang for r in rl])
            )
    corrupted = {}
    for cond in sorted({r["condition"] for r in rows} - {"clean"}):
        rs = [r for r in rows if r["condition"] == cond]
        corrupted[cond] = float(np.mean([r["full"]["pred"] == r["language"] for r in rs]))
    confusions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["condition"] == "clean":
            for crop in CROPS:
                if r[crop]["pred"] != r["language"]:
                    confusions[f"{r['language']}@{crop}"][r[crop]["pred"]] += 1
    crop_time = {c: sum(r[c]["time_s"] for r in rows if c in r) for c in CROPS}
    crop_audio = {
        c: sum(min(e["duration_s"], s or 1e9) for e in entries if e["group"] == "clean")
        for c, s in CROPS.items()
    }
    metrics = {
        "accuracy_clean": acc,
        "accuracy_full_by_corruption": corrupted,
        "confusions": {k: dict(v) for k, v in confusions.items()},
        "rtf_by_crop": {c: crop_time[c] / crop_audio[c] for c in CROPS},
        "ms_per_call_by_crop": {c: 1000 * crop_time[c] / sum(c in r for r in rows) for c in CROPS},
        "peak_rss_mb": peak_rss_mb(),
        "mlx_peak_mb": mlx_peak_mb(),
    }
    return {
        "config": config,
        "load_s": load_s,
        "compute_s": compute,
        "audio_s": audio_s,
        "rtf": compute / audio_s,
        "metrics": metrics,
        "per_clip": rows,
    }


# ---------------------------------------------------------------------------- ASR


def _asr_backend(candidate: str) -> tuple[Callable[[np.ndarray, Path], dict], dict[str, Any]]:
    repo = ASR_REPOS[candidate]
    if candidate.startswith("whisper_turbo"):
        words_on = candidate == "whisper_turbo"
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.get_model(repo, mx.float16)

        def run(audio: np.ndarray, _: Path) -> dict[str, Any]:
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo=repo, word_timestamps=words_on, verbose=None
            )
            segs = r["segments"]
            words = [w for s in segs for w in s.get("words", [])]
            return {
                "text": r["text"],
                "language": r["language"],
                "signals": {
                    "no_speech_prob": [round(s["no_speech_prob"], 4) for s in segs],
                    "avg_logprob": [round(s["avg_logprob"], 4) for s in segs],
                    "compression_ratio": [round(s["compression_ratio"], 3) for s in segs],
                    "temperature": [s["temperature"] for s in segs],
                    "word_prob_mean": float(np.mean([w["probability"] for w in words]))
                    if words
                    else None,
                    "first_word_start": words[0]["start"] if words else None,
                    "last_word_end": words[-1]["end"] if words else None,
                },
            }

        exposes = {
            "word_timestamps": words_on,
            "no_speech_prob": True,
            "avg_logprob": True,
            "token_confidences": "word probability",
            "compression_ratio": True,
            "language": "detected language + probabilities",
        }
        cfg = {
            "runtime": "mlx-whisper",
            "dtype": "float16",
            "decode": "transcribe() defaults: language auto-detect, temperature fallback "
            f"(0.0..1.0), condition_on_previous_text=True, word_timestamps={words_on}",
        }
    elif candidate == "parakeet_v3":
        from parakeet_mlx import from_pretrained

        model = from_pretrained(repo)

        def run(_: np.ndarray, path: Path) -> dict[str, Any]:
            r = model.transcribe(path)
            toks = r.tokens
            return {
                "text": r.text,
                "language": None,
                "signals": {
                    "token_conf_mean": float(np.mean([t.confidence for t in toks]))
                    if toks
                    else None,
                    "token_conf_min": float(np.min([t.confidence for t in toks])) if toks else None,
                    "sentence_conf": [round(s.confidence, 4) for s in r.sentences],
                    "first_token_start": toks[0].start if toks else None,
                    "last_token_end": toks[-1].end if toks else None,
                },
            }

        exposes = {
            "word_timestamps": "token/word timestamps",
            "no_speech_prob": False,
            "avg_logprob": False,
            "token_confidences": True,
            "language": False,
        }
        cfg = {"runtime": "parakeet-mlx", "dtype": "bfloat16", "decode": "greedy TDT, no chunking"}
    else:
        from mlx_audio.stt.utils import load_model

        model = load_model(repo)

        def run(audio: np.ndarray, _: Path) -> dict[str, Any]:
            tokens, lps = [], []
            for tok, logprobs in model.stream_generate(audio, max_tokens=QWEN_MAX_TOKENS):
                tokens.append(int(tok))
                lps.append(float(logprobs[int(tok)]))
            raw = model._tokenizer.decode(tokens, skip_special_tokens=True)
            lang, text = model.extract_language(raw)
            return {
                "text": text,
                "language": lang,
                "signals": {
                    "n_tokens": len(tokens),
                    "hit_max_tokens": len(tokens) >= QWEN_MAX_TOKENS,
                    "token_logprob_mean": float(np.mean(lps)) if lps else None,
                    "token_logprob_min": float(np.min(lps)) if lps else None,
                },
            }

        exposes = {
            "word_timestamps": "no (separate Qwen3-ForcedAligner model)",
            "no_speech_prob": "no ('language None' tag on no speech)",
            "avg_logprob": "computable from stream_generate logprobs",
            "token_confidences": "token logprobs via stream_generate",
            "language": "detected language name in output",
        }
        cfg = {
            "runtime": "mlx-audio",
            "dtype": "8-bit quantized (mlx-community)",
            "decode": f"greedy, language auto-detect, max_tokens={QWEN_MAX_TOKENS}",
        }
    cfg.update({"weights": repo, "upstream": ASR_UPSTREAM[candidate], "exposes": exposes})
    return run, cfg


def bench_asr(candidate: str) -> dict[str, Any]:
    import jiwer

    entries = load_manifest()
    t0 = time.perf_counter()
    run, config = _asr_backend(candidate)
    load_s = time.perf_counter() - t0
    warm = next(e for e in entries if e["condition"] == "clean")
    run(load_audio(warm), CLIPS / warm["path"])  # warm-up, not timed

    def per_clip(e: dict[str, Any], audio: np.ndarray) -> dict[str, Any]:
        out = run(audio, CLIPS / e["path"])
        hyp = normalize_text(out["text"])
        out.update(
            {
                "hyp_norm": hyp,
                "ref_norm": normalize_text(e["reference"]),
                "language_true": e["language"],
                "n_words": len(hyp.split()),
            }
        )
        return out

    rows, compute, audio_s = run_over_clips(entries, per_clip, f"asr/{candidate}")

    def wer(rs: list[dict[str, Any]]) -> dict[str, Any] | None:
        rs = [r for r in rs if r["ref_norm"]]
        if not rs:
            return None
        o = jiwer.process_words([r["ref_norm"] for r in rs], [r["hyp_norm"] or "" for r in rs])
        n_ref = sum(len(r["ref_norm"].split()) for r in rs)
        return {
            "wer": o.wer,
            "n": len(rs),
            "ref_words": n_ref,
            "sub": o.substitutions,
            "del": o.deletions,
            "ins": o.insertions,
        }

    by_cond_lang: dict[str, Any] = {}
    for cond in sorted({r["condition"] for r in rows if r["ref_norm"]}):
        for lang in ("en", "ru", "he"):
            rs = [r for r in rows if r["condition"] == cond and r["language_true"] == lang]
            if rs:
                by_cond_lang[f"{cond}/{lang}"] = wer(rs)
    # Paired view: the corruption sources' clean versions vs every corruption.
    nonspeech = [r for r in rows if r["condition"].startswith("only_")]
    halluc = {}
    for cond in sorted({r["condition"] for r in nonspeech}):
        rs = [r for r in nonspeech if r["condition"] == cond]
        halluc[cond] = {
            "n": len(rs),
            "clips_with_output": sum(r["n_words"] > 0 for r in rs),
            "words": sum(r["n_words"] for r in rs),
            "examples": [r["hyp_norm"][:80] for r in rs if r["n_words"]][:3],
        }
    lang_rows = [r for r in rows if r["condition"] == "clean" and r["language"] is not None]
    metrics = {
        "wer_by_condition_language": by_cond_lang,
        "wer_clean_en": by_cond_lang.get("clean/en"),
        "wer_clean_ru": by_cond_lang.get("clean/ru"),
        "wer_clean_he": by_cond_lang.get("clean/he"),
        "hallucination_nonspeech": halluc,
        "hallucination_words_total": sum(r["n_words"] for r in nonspeech),
        "hallucination_clips_with_output": sum(r["n_words"] > 0 for r in nonspeech),
        "detected_language_clean": {
            lang: dict(
                sorted(
                    _count(r["language"] for r in lang_rows if r["language_true"] == lang).items()
                )
            )
            for lang in ("en", "ru", "he")
        }
        if lang_rows
        else None,
        "peak_rss_mb": peak_rss_mb(),
        "mlx_peak_mb": mlx_peak_mb(),
    }
    return {
        "config": config,
        "load_s": load_s,
        "compute_s": compute,
        "audio_s": audio_s,
        "rtf": compute / audio_s,
        "metrics": metrics,
        "per_clip": rows,
    }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for i in items:
        out[str(i)] += 1
    return dict(out)


# ---------------------------------------------------------------------------- driver

BENCHES = {"vad": bench_vad, "tagger": bench_tagger, "lid": bench_lid, "asr": bench_asr}


def ensure_efficientat() -> None:
    if not (EFFICIENTAT_DIR / ".git").exists():
        EFFICIENTAT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", EFFICIENTAT_REPO, str(EFFICIENTAT_DIR)], check=True)
    subprocess.run(
        ["git", "-C", str(EFFICIENTAT_DIR), "checkout", "-q", EFFICIENTAT_COMMIT], check=True
    )


def run_one(component: str, candidate: str) -> None:
    if component == "tagger":
        ensure_efficientat()
    t0 = time.perf_counter()
    result = BENCHES[component](candidate)
    result.update(
        {
            "component": component,
            "candidate": candidate,
            "peak_rss_mb": peak_rss_mb(),
            "mlx_peak_mb": mlx_peak_mb(),
            "wall_s": time.perf_counter() - t0,
            "env": env_info(),
        }
    )
    out = RESULTS_DIR / f"{component}__{candidate}.json"
    write_json(out, result)
    print(
        f"wrote {out} (rtf={result['rtf']:.4f}, load={result['load_s']:.1f}s, "
        f"peak_rss={result['peak_rss_mb']:.0f} MB)"
    )


def run_all(components: list[str]) -> None:
    failures = {}
    for comp in components:
        for cand in CANDIDATES[comp]:
            print(f"=== {comp} / {cand}", flush=True)
            proc = subprocess.run([sys.executable, __file__, "run", comp, cand])
            if proc.returncode:
                failures[f"{comp}/{cand}"] = proc.returncode
    write_json(RESULTS_DIR / "failures.json", failures)
    import report

    report.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("component", choices=list(CANDIDATES))
    p_run.add_argument("candidate")
    p_all = sub.add_parser("all")
    p_all.add_argument("--only", nargs="*", choices=list(CANDIDATES), default=list(CANDIDATES))
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.cmd == "run":
        if args.candidate not in CANDIDATES[args.component]:
            parser.error(f"candidate must be one of {CANDIDATES[args.component]}")
        run_one(args.component, args.candidate)
    elif args.cmd == "all":
        run_all(args.only)
    else:
        import report

        report.main()


if __name__ == "__main__":
    main()
