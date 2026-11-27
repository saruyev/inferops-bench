"""Render results/summary.md from results/*.json (called by `bench.py all` / `bench.py summary`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import DATASETS, RESULTS_DIR

RECOMMENDATIONS = Path(__file__).with_name("recommendations.md")

# Licenses as stated by each source (model card / repository LICENSE / PyPI metadata).
LICENSES = [
    ("Silero VAD v6", "silero-vad (PyPI) / snakers4/silero-vad", "MIT", "MIT (bundled weights)"),
    (
        "WebRTC VAD",
        "webrtcvad-wheels (PyPI)",
        "MIT (wrapper); WebRTC code BSD-3-Clause",
        "n/a (no learned weights)",
    ),
    (
        "EfficientAT mn10_as",
        "github.com/fschmid56/EfficientAT",
        "MIT",
        "MIT (GitHub release v0.0.1, same repository); trained on AudioSet",
    ),
    (
        "AST AudioSet",
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        "transformers: Apache-2.0",
        "BSD-3-Clause (model card)",
    ),
    (
        "SpeechBrain VoxLingua107 ECAPA",
        "speechbrain/lang-id-voxlingua107-ecapa",
        "speechbrain: Apache-2.0",
        "Apache-2.0 (model card)",
    ),
    (
        "Whisper large-v3-turbo",
        "openai/whisper-large-v3-turbo -> mlx-community port",
        "mlx-whisper: MIT",
        "MIT (model card)",
    ),
    (
        "Parakeet TDT 0.6B v3",
        "nvidia/parakeet-tdt-0.6b-v3 -> mlx-community port",
        "parakeet-mlx: Apache-2.0",
        "CC-BY-4.0 (model card)",
    ),
    (
        "Qwen3-ASR 1.7B / 0.6B",
        "Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B -> mlx-community 8-bit",
        "mlx-audio: MIT",
        "Apache-2.0 (model card)",
    ),
]


def _load() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(RESULTS_DIR.glob("*__*.json")):
        d = json.loads(p.read_text())
        out[f"{d['component']}/{d['candidate']}"] = d
    return out


def _f(x: Any, nd: int = 3, pct: bool = False) -> str:
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def vad_section(res: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, d in res.items():
        if not key.startswith("vad/"):
            continue
        m, bc = d["metrics"], d["metrics"]["by_condition"]
        rows.append(
            [
                d["candidate"],
                _f(d["rtf"], 5),
                _f(m["mean_ratio_error_speech_clips"]),
                _f(m["missed_speech_frac_clean"], pct=True),
                _f(bc["music_snr0"]["missed_speech_frac"], pct=True),
                _f(bc["noise_snr0"]["missed_speech_frac"], pct=True),
                _f(m["nonspeech_music"]["false_speech_frac"], pct=True),
                _f(m["nonspeech_noise"]["false_speech_frac"], pct=True),
                _f(m["nonspeech_silence"]["false_speech_frac"], pct=True),
                _f(bc["pad_zero"]["false_speech_s_per_min"], 2),
                _f(bc["pad_floor"]["false_speech_s_per_min"], 2),
            ]
        )
    head = [
        "candidate",
        "RTF (1 thread)",
        "speech-ratio err (speech clips)",
        "missed, clean",
        "missed, music 0 dB",
        "missed, noise 0 dB",
        "false speech, music-only",
        "false speech, noise-only",
        "false speech, silence",
        "false s/min in zero pad",
        "false s/min in floor pad",
    ]
    return _table(head, rows)


def _rates(block: dict[str, Any]) -> str:
    return f"{_f(block['tpr_at_0.5'], 2)} / {_f(block['fpr_at_0.5'], 2)}"


def tagger_section(res: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, d in res.items():
        if not key.startswith("tagger/"):
            continue
        m = d["metrics"]
        rows.append(
            [
                d["candidate"],
                d["config"]["device"],
                _f(m["music_present"]["auc"]),
                _f(m["noise_present"]["auc"]),
                _f(m["speech_present"]["auc"]),
                _rates(m["music_present"]),
                _rates(m["noise_present"]),
                _f(d["rtf"], 4),
                _f(d["load_s"], 1),
                f"{d['peak_rss_mb']:.0f}",
            ]
        )
    head = [
        "candidate",
        "device",
        "music AUC",
        "noise AUC",
        "speech AUC",
        "music TPR/FPR @0.5",
        "noise TPR/FPR @0.5",
        "RTF",
        "load s",
        "peak RSS MB",
    ]
    return _table(head, rows)


def lid_section(res: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, d in res.items():
        if not key.startswith("lid/"):
            continue
        a = d["metrics"]["accuracy_clean"]
        ms = d["metrics"]["ms_per_call_by_crop"]
        rows.append(
            [d["candidate"]]
            + [
                " / ".join(_f(a[c][lang], pct=True) for lang in ("en", "ru", "he"))
                for c in ("1s", "3s", "full")
            ]
            + [
                " / ".join(
                    _f(a[c].get("all_restricted_en_ru_he"), pct=True) for c in ("1s", "3s", "full")
                ),
                " / ".join(f"{ms[c]:.0f}" for c in ("1s", "3s", "full")),
                _f(d["load_s"], 1),
                f"{d['peak_rss_mb']:.0f}",
            ]
        )
    head = [
        "candidate",
        "acc 1 s (en/ru/he)",
        "acc 3 s (en/ru/he)",
        "acc full (en/ru/he)",
        "acc restricted to {en,ru,he}, 1s/3s/full",
        "ms/call 1s/3s/full",
        "load s",
        "peak RSS MB",
    ]
    out = _table(head, rows)
    corr = [
        [d["candidate"]]
        + [_f(v, pct=True) for v in d["metrics"]["accuracy_full_by_corruption"].values()]
        for k, d in res.items()
        if k.startswith("lid/")
    ]
    if corr:
        conds = list(
            next(d for k, d in res.items() if k.startswith("lid/"))["metrics"][
                "accuracy_full_by_corruption"
            ]
        )
        out += "\n\nFull-clip accuracy on corrupted copies (en+ru, 10 clips each):\n\n"
        out += _table(["candidate", *conds], corr)
    return out


def asr_section(res: dict[str, dict[str, Any]]) -> str:
    rows, corr_rows, sig_rows = [], [], []
    conds = [
        "clean",
        "pad_zero",
        "pad_floor",
        "music_snr10",
        "music_snr0",
        "noise_snr10",
        "noise_snr5",
        "noise_snr0",
    ]
    for key, d in res.items():
        if not key.startswith("asr/"):
            continue
        m = d["metrics"]
        w = m["wer_by_condition_language"]
        rows.append(
            [
                d["candidate"],
                d["config"]["runtime"],
                d["config"]["dtype"],
                _f((m["wer_clean_en"] or {}).get("wer"), pct=True),
                _f((m["wer_clean_ru"] or {}).get("wer"), pct=True),
                _f((m["wer_clean_he"] or {}).get("wer"), pct=True),
                f"{m['hallucination_words_total']} / {m['hallucination_clips_with_output']} of 24",
                _f(d["rtf"], 4),
                _f(d["load_s"], 1),
                f"{d['peak_rss_mb']:.0f}",
                _f(d.get("mlx_peak_mb"), 0),
            ]
        )
        # Paired corruption view: only the 10 corruption sources (7 en + 3 ru).
        src = {r["clip_id"]: r for r in d["per_clip"]}
        pc = []
        for cond in conds:
            if cond == "clean":
                ids = _corruption_source_clean_ids(d)
            else:
                ids = [cid for cid, r in src.items() if r["condition"] == cond]
            pc.append(_f(_pooled_wer([src[i] for i in ids]), pct=True))
        corr_rows.append([d["candidate"], *pc])
        e = d["config"]["exposes"]
        sig_rows.append(
            [d["candidate"]]
            + [
                str(e.get(k, "-"))
                for k in (
                    "word_timestamps",
                    "no_speech_prob",
                    "avg_logprob",
                    "token_confidences",
                    "language",
                )
            ]
        )
        del w
    head = [
        "candidate",
        "runtime",
        "weights",
        "WER clean en",
        "WER clean ru",
        "WER clean he*",
        "non-speech words / clips with output",
        "RTF",
        "load s",
        "peak RSS MB",
        "MLX peak MB",
    ]
    out = _table(head, rows)
    out += (
        "\n\n\\* Hebrew is outside the supported language list of Parakeet v3 and "
        "Qwen3-ASR; it is there to show unsupported-language behaviour.\n\n"
    )
    out += "WER on the 10 corruption sources (7 en + 3 ru), same utterances in every column:\n\n"
    out += _table(["candidate", *conds], corr_rows)
    out += "\n\nSignals each runtime exposes:\n\n"
    out += _table(
        [
            "candidate",
            "word timestamps",
            "no_speech_prob",
            "avg_logprob",
            "token confidences",
            "language",
        ],
        sig_rows,
    )
    return out


def _corruption_source_clean_ids(d: dict[str, Any]) -> list[str]:
    ds = json.loads((RESULTS_DIR / "dataset.json").read_text())
    wanted = set(ds["params"]["corruption_sources"])
    from common import load_manifest

    return [
        e["clip_id"]
        for e in load_manifest()
        if e["condition"] == "clean" and e["source_id"] in wanted
    ]


def _pooled_wer(rows: list[dict[str, Any]]) -> float | None:
    import jiwer

    rows = [r for r in rows if r["ref_norm"]]
    if not rows:
        return None
    return jiwer.wer([r["ref_norm"] for r in rows], [r["hyp_norm"] for r in rows])


SIGNALS = {
    "max no_speech_prob": lambda s: max(s["no_speech_prob"], default=None),
    "-mean avg_logprob": lambda s: (
        -sum(s["avg_logprob"]) / len(s["avg_logprob"]) if s.get("avg_logprob") else None
    ),
    "-word_prob_mean": lambda s: -s["word_prob_mean"] if s.get("word_prob_mean") else None,
    "-token_conf_mean": lambda s: -s["token_conf_mean"] if s.get("token_conf_mean") else None,
    "-token_logprob_mean": lambda s: (
        -s["token_logprob_mean"] if s.get("token_logprob_mean") is not None else None
    ),
}


def signal_separability(res: dict[str, dict[str, Any]]) -> str:
    """AUC of each confidence signal for 'this non-empty output came from a non-speech clip'."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for key, d in res.items():
        if not key.startswith("asr/"):
            continue
        clips = [r for r in d["per_clip"] if r["n_words"] > 0]
        y = [r["condition"].startswith("only_") for r in clips]
        for name, fn in SIGNALS.items():
            try:
                vals = [fn(r["signals"]) for r in clips]
            except (KeyError, TypeError):
                continue
            pairs = [(v, t) for v, t in zip(vals, y, strict=True) if v is not None and v == v]
            if len(pairs) < len(clips) * 0.9 or len({t for _, t in pairs}) < 2:
                continue
            auc = roc_auc_score([t for _, t in pairs], [v for v, _ in pairs])
            rows.append([d["candidate"], name, _f(auc), f"{sum(y)} / {len(clips)}"])
    return _table(["candidate", "signal", "AUC", "non-speech / all clips with output"], rows)


def hallucination_examples(res: dict[str, dict[str, Any]]) -> str:
    lines = []
    for key, d in res.items():
        if not key.startswith("asr/"):
            continue
        for cond, h in d["metrics"]["hallucination_nonspeech"].items():
            for ex in h["examples"][:2]:
                lines.append(f'- {d["candidate"]} on `{cond}`: "{ex}"')
    return "\n".join(lines) if lines else "- none"


def main() -> None:
    res = _load()
    ds = json.loads((RESULTS_DIR / "dataset.json").read_text())
    env = next(iter(res.values()))["env"] if res else {}
    failures_path = RESULTS_DIR / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
    parts = [
        "# Model-choice spike: VAD, audio tagger, LID, local ASR",
        "",
        f"Machine: {env.get('cpu', '?')}, {env.get('mem_gb', 0):.0f} GB, {env.get('platform')}; "
        f"Python {env.get('python')}. Package versions are in each results/*.json (`env`).",
        "",
        f"Test set: {ds['n_clips']} clips, {ds['total_audio_s'] / 60:.1f} min of 16 kHz mono "
        f"audio, seed {ds['seed']}. Clips per condition: "
        + ", ".join(f"{k} {v}" for k, v in ds["clips_per_condition"].items())
        + ".",
        "",
        f"Ground truth: {ds['ground_truth']}",
        "",
        "RTF = compute seconds / audio seconds, model load and one warm-up call excluded. "
        "Peak RSS is the process maximum (Python + framework + model); for MLX models the "
        "Metal allocator peak is listed separately because unified-memory buffers are not "
        "fully visible in RSS.",
        "",
        "## Recommendations",
        "",
        RECOMMENDATIONS.read_text().strip() if RECOMMENDATIONS.exists() else "_pending_",
        "",
        "## VAD",
        "",
        vad_section(res),
        "",
        "Speech-ratio error = |predicted speech fraction - true speech fraction| per clip. Ground "
        "truth counts the whole utterance as speech, so pauses inside an utterance count as "
        "'missed' for a VAD that (correctly) closes segments on pauses; compare the 'missed' "
        "columns between candidates rather than reading them as absolute.",
        "",
        "## Audio tagger",
        "",
        tagger_section(res),
        "",
        "Class mapping (AudioSet ontology subtrees, 527 classes) is in "
        "`results/audioset_class_mapping.json`. Positives: music bed (SNR 0/10 dB) and "
        "music-only clips for music; noise bed (SNR 0/5/10 dB) and noise-only clips for noise. "
        "The -60 dBFS floor and digital silence count as negatives.",
        "",
        "## Language ID",
        "",
        lid_section(res),
        "",
        "Crops are centered in the clean utterance. Accuracy columns are open-set top-1 (107 "
        "languages for SpeechBrain, 100 for Whisper); the restricted column takes the argmax over "
        "en/ru/he only, i.e. what a pipeline that knows its expected languages would see. "
        "Whisper always encodes a 30 s window, so its cost per call does not shrink with the crop.",
        "",
        "## Local ASR",
        "",
        asr_section(res),
        "",
        "How well each exposed signal flags output that came from a non-speech clip (AUC over "
        "clips with any output; 1.0 = a threshold on the signal would drop every hallucination "
        "and keep every real transcript):",
        "",
        signal_separability(res),
        "",
        "Hallucination examples on non-speech clips (normalized text, first 80 chars):",
        "",
        hallucination_examples(res),
        "",
        "## Licenses",
        "",
        _table(
            ["component", "source", "code license", "weights license"], [list(r) for r in LICENSES]
        ),
        "",
        "Datasets:",
        "",
        _table(
            ["dataset", "url used", "license (as stated)", "note"],
            [[k, v["url"], v["license"], v.get("note", "")] for k, v in DATASETS.items()],
        ),
        "",
        "## Failures and skips",
        "",
        "\n".join(f"- `{k}` exited with code {v}" for k, v in failures.items())
        or "- No candidate run failed.",
        "",
    ]
    skips = Path(__file__).with_name("skips.md")
    if skips.exists():
        parts += [skips.read_text().strip(), ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(parts))
    print(f"wrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
