"""Build the ~250-clip spike test set (16 kHz mono) and its ground-truth manifest.

Usage: uv run python prepare.py [--force]

Clips and the manifest go to $PARIBUS_SPIKE_CACHE/clips (default ~/.cache/paribus-spike/clips).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from common import (
    CLIPS,
    DATASETS,
    MANIFEST,
    RESULTS_DIR,
    SEED,
    SR,
    download,
    fetch,
    write_json,
)

N_LIBRI, N_RU, N_HE = 100, 50, 20
MIN_S, MAX_S = 3.0, 20.0
# Corrupted copies: the same source utterances go through every corruption (paired design).
N_CORRUPT_EN, N_CORRUPT_RU = 7, 3
FLOOR_DBFS = -60.0  # RMS of the "low noise floor" (white Gaussian noise)
N_MUSIC_ONLY, N_NOISE_ONLY, N_ZERO_ONLY, N_FLOOR_ONLY = 8, 8, 4, 4


def _decode(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    return audio.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def _floor(rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.standard_normal(n) * 10 ** (FLOOR_DBFS / 20)).astype(np.float32)


def _segment(rng: np.random.Generator, source: np.ndarray, n: int) -> np.ndarray:
    """Random n-sample excerpt of ``source``; tiled when the source is shorter."""
    if len(source) < n:
        source = np.tile(source, int(np.ceil(n / len(source))) + 1)
    start = int(rng.integers(0, len(source) - n + 1))
    return source[start : start + n].copy()


def _mix(speech: np.ndarray, bed: np.ndarray, snr_db: float) -> np.ndarray:
    gain = _rms(speech) / (_rms(bed) * 10 ** (snr_db / 20))
    return speech + gain * bed


def _peak_safe(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (0.98 / peak)).astype(np.float32) if peak > 0.98 else x.astype(np.float32)


def load_librispeech(rng: np.random.Generator) -> list[dict[str, Any]]:
    path = fetch("librispeech_test_clean")
    table = pq.read_table(path, columns=["id", "text", "audio"])
    ids, texts, audios = (table.column(c).to_pylist() for c in ("id", "text", "audio"))
    out: list[dict[str, Any]] = []
    for i in rng.permutation(len(ids)):
        audio = _decode(audios[i]["bytes"])
        if MIN_S <= len(audio) / SR <= MAX_S:
            out.append(
                {
                    "source_id": f"librispeech:{ids[i]}",
                    "language": "en",
                    "reference": texts[i],
                    "audio": audio,
                    "dataset": "librispeech_test_clean",
                }
            )
        if len(out) == N_LIBRI:
            break
    return out


def load_fleurs(rng: np.random.Generator, name: str, lang: str, n: int) -> list[dict[str, Any]]:
    tar_path = fetch(name)
    tsv = download(DATASETS[name]["tsv"], tar_path.parent / "test.tsv")
    rows = {}
    with tsv.open(newline="") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
            rows[row[1]] = row
    names = sorted(rows)
    order = [names[i] for i in rng.permutation(len(names))]
    wanted = {fn: k for k, fn in enumerate(order)}
    found: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            base = Path(member.name).name
            if member.isfile() and base in wanted:
                fh = tar.extractfile(member)
                if fh is not None:
                    found[base] = _decode(fh.read())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()  # FLEURS has several speakers per sentence; keep one each
    for fn in order:
        audio = found.get(fn)
        row = rows[fn]
        if audio is None or row[0] in seen or not MIN_S <= len(audio) / SR <= MAX_S:
            continue
        seen.add(row[0])
        out.append(
            {
                "source_id": f"fleurs_{lang}:{row[0]}:{fn}",
                "language": lang,
                "reference": row[2],
                "audio": audio,
                "dataset": name,
            }
        )
        if len(out) == n:
            break
    return out


def load_backgrounds() -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    music, noise = [], []
    for name, bucket in (
        ("musan_music_cc_by_3", music),
        ("musan_music_cc_by_sa_3", music),
        ("musan_noise", noise),
    ):
        table = pq.read_table(fetch(name), columns=["sample_id", "audio", "audio_license"])
        for sid, audio, lic in zip(
            *(table.column(c).to_pylist() for c in ("sample_id", "audio", "audio_license")),
            strict=True,
        ):
            wav = _decode(audio["bytes"])
            if len(wav) >= 2 * SR and _rms(wav) > 1e-4:
                bucket.append((f"{sid} ({lic})", wav))
    return sorted(music, key=lambda t: t[0]), sorted(noise, key=lambda t: t[0])


def build(force: bool) -> None:
    if MANIFEST.exists() and not force:
        print(f"{MANIFEST} exists; use --force to rebuild")
        return
    if CLIPS.exists():
        shutil.rmtree(CLIPS)
    (CLIPS / "wav").mkdir(parents=True)
    fetch("audioset_ontology")
    rng = np.random.default_rng(SEED)

    en = load_librispeech(rng)
    ru = load_fleurs(rng, "fleurs_ru_test", "ru", N_RU)
    he = load_fleurs(rng, "fleurs_he_test", "he", N_HE)
    music, noise = load_backgrounds()
    print(f"speech: en={len(en)} ru={len(ru)} he={len(he)}; music={len(music)} noise={len(noise)}")

    entries: list[dict[str, Any]] = []

    def emit(clip_id: str, audio: np.ndarray, **meta: Any) -> None:
        audio = _peak_safe(audio)
        path = CLIPS / "wav" / f"{clip_id}.wav"
        sf.write(path, audio, SR, subtype="PCM_16")
        dur = len(audio) / SR
        segs = meta.pop("speech_segments", [])
        entries.append(
            {
                "clip_id": clip_id,
                "path": str(path.relative_to(CLIPS)),
                "duration_s": dur,
                "speech_segments": segs,
                "speech_s": sum(e - s for s, e in segs),
                "has_speech": bool(segs),
                **meta,
            }
        )

    for k, src in enumerate(en + ru + he):
        dur = len(src["audio"]) / SR
        emit(
            f"clean_{src['language']}_{k:03d}",
            src["audio"],
            condition="clean",
            group="clean",
            language=src["language"],
            reference=src["reference"],
            source_id=src["source_id"],
            dataset=src["dataset"],
            speech_segments=[[0.0, dur]],
            music_present=False,
            noise_present=False,
        )

    sources = en[:N_CORRUPT_EN] + ru[:N_CORRUPT_RU]
    for k, src in enumerate(sources):
        speech = src["audio"]
        dur = len(speech) / SR
        base = {
            "group": "corrupted",
            "language": src["language"],
            "reference": src["reference"],
            "source_id": src["source_id"],
            "dataset": src["dataset"],
        }
        for kind in ("zero", "floor"):
            lead, tail = (float(x) for x in rng.uniform(5.0, 20.0, size=2))
            n_lead, n_tail = int(lead * SR), int(tail * SR)
            audio = np.concatenate(
                [np.zeros(n_lead, np.float32), speech, np.zeros(n_tail, np.float32)]
            )
            if kind == "floor":
                audio = audio + _floor(rng, len(audio))
            t0 = n_lead / SR
            emit(
                f"pad_{kind}_{k:02d}",
                audio,
                condition=f"pad_{kind}",
                **base,
                speech_segments=[[t0, t0 + dur]],
                pad_lead_s=t0,
                pad_tail_s=n_tail / SR,
                music_present=False,
                noise_present=False,
            )
        for snr in (0, 10):
            bg_id, bg = music[int(rng.integers(len(music)))]
            emit(
                f"music_snr{snr}_{k:02d}",
                _mix(speech, _segment(rng, bg, len(speech)), snr),
                condition=f"music_snr{snr}",
                **base,
                snr_db=snr,
                background=bg_id,
                speech_segments=[[0.0, dur]],
                music_present=True,
                noise_present=False,
            )
        for snr in (0, 5, 10):
            bg_id, bg = noise[int(rng.integers(len(noise)))]
            emit(
                f"noise_snr{snr}_{k:02d}",
                _mix(speech, _segment(rng, bg, len(speech)), snr),
                condition=f"noise_snr{snr}",
                **base,
                snr_db=snr,
                background=bg_id,
                speech_segments=[[0.0, dur]],
                music_present=False,
                noise_present=True,
            )

    def nonspeech(prefix: str, n: int, make: Any, **meta: Any) -> None:
        for k in range(n):
            dur = float(rng.uniform(5.0, 30.0))
            audio, bg_id = make(int(dur * SR))
            emit(
                f"{prefix}_{k:02d}",
                audio,
                group="nonspeech",
                language=None,
                reference="",
                source_id=None,
                background=bg_id,
                **meta,
            )

    def pick(pool: list[tuple[str, np.ndarray]]) -> Any:
        def make(n: int) -> tuple[np.ndarray, str]:
            bg_id, bg = pool[int(rng.integers(len(pool)))]
            seg = _segment(rng, bg, n)
            return seg * (10 ** (-20 / 20) / _rms(seg)), bg_id  # normalize to -20 dBFS RMS

        return make

    nonspeech(
        "only_music",
        N_MUSIC_ONLY,
        pick(music),
        condition="only_music",
        music_present=True,
        noise_present=False,
    )
    nonspeech(
        "only_noise",
        N_NOISE_ONLY,
        pick(noise),
        condition="only_noise",
        music_present=False,
        noise_present=True,
    )
    nonspeech(
        "only_zero",
        N_ZERO_ONLY,
        lambda n: (np.zeros(n, np.float32), None),
        condition="only_zero",
        music_present=False,
        noise_present=False,
    )
    nonspeech(
        "only_floor",
        N_FLOOR_ONLY,
        lambda n: (_floor(rng, n), None),
        condition="only_floor",
        music_present=False,
        noise_present=False,
    )

    with MANIFEST.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_cond: dict[str, int] = {}
    for e in entries:
        by_cond[e["condition"]] = by_cond.get(e["condition"], 0) + 1
    write_json(
        RESULTS_DIR / "dataset.json",
        {
            "seed": SEED,
            "sample_rate": SR,
            "n_clips": len(entries),
            "clips_per_condition": by_cond,
            "total_audio_s": sum(e["duration_s"] for e in entries),
            "params": {
                "speech_duration_s": [MIN_S, MAX_S],
                "noise_floor_dbfs_rms": FLOOR_DBFS,
                "pad_lead_tail_s": [5.0, 20.0],
                "nonspeech_duration_s": [5.0, 30.0],
                "nonspeech_level_dbfs_rms": -20.0,
                "corruption_sources": [s["source_id"] for s in sources],
            },
            "ground_truth": "Original utterances count as speech end to end (internal pauses "
            "included); padded regions, music-only, noise-only and silence clips "
            "are non-speech. music_present / noise_present mark an added music or "
            "noise bed; the -60 dBFS noise floor is not counted as noise.",
            "sources": DATASETS,
        },
    )
    print(f"wrote {len(entries)} clips to {CLIPS}: {by_cond}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild clips and manifest")
    build(parser.parse_args().force)


if __name__ == "__main__":
    main()
