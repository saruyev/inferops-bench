"""Shared paths, dataset sources and small helpers for the model-choice spike."""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

SPIKE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SPIKE_DIR / "results"
CACHE = Path(os.environ.get("PARIBUS_SPIKE_CACHE", Path.home() / ".cache" / "paribus-spike"))
RAW = CACHE / "raw"
CLIPS = CACHE / "clips"
MANIFEST = CLIPS / "manifest.jsonl"
HF_HOME = CACHE / "hf"
SR = 16_000
SEED = 20261127

# Keep every Hugging Face / torch hub download inside the spike cache.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))

_HF = "https://huggingface.co/datasets"
_AKIKI = f"{_HF}/Akiki-AI/akiki-vad-silero623-v1"

# Every dataset we touch, with the license as stated by the source.
DATASETS: dict[str, dict[str, Any]] = {
    "librispeech_test_clean": {
        "url": f"{_HF}/openslr/librispeech_asr/resolve/main/clean/test/0000.parquet",
        "local": "librispeech/test-clean.parquet",
        "homepage": "https://www.openslr.org/12/",
        "mirror": "https://huggingface.co/datasets/openslr/librispeech_asr",
        "license": "CC-BY-4.0",
        "note": "openslr.org mirror ran at ~80 KB/s, so the identical HF parquet was used.",
    },
    "fleurs_ru_test": {
        "url": f"{_HF}/google/fleurs/resolve/main/data/ru_ru/audio/test.tar.gz",
        "tsv": f"{_HF}/google/fleurs/resolve/main/data/ru_ru/test.tsv",
        "local": "fleurs/ru_ru/test.tar.gz",
        "homepage": "https://huggingface.co/datasets/google/fleurs",
        "license": "CC-BY-4.0",
    },
    "fleurs_he_test": {
        "url": f"{_HF}/google/fleurs/resolve/main/data/he_il/audio/test.tar.gz",
        "tsv": f"{_HF}/google/fleurs/resolve/main/data/he_il/test.tsv",
        "local": "fleurs/he_il/test.tar.gz",
        "homepage": "https://huggingface.co/datasets/google/fleurs",
        "license": "CC-BY-4.0",
    },
    "musan_noise": {
        "url": f"{_AKIKI}-musan-noise/resolve/main/data/musan_noise/train-00000.parquet",
        "local": "musan_hf/noise/train.parquet",
        "homepage": "https://www.openslr.org/17/",
        "mirror": "https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-noise",
        "license": "CC-BY-4.0 (per-recording audio_license field; MUSAN noise subset)",
        "note": "Partial MUSAN mirror (104 of 930 noise recordings, original audio, 16 kHz). "
        "Full MUSAN is an 11 GB tarball; only the audio column is used.",
    },
    "musan_music_cc_by_3": {
        "url": f"{_AKIKI}-musan-music-cc-by-3.0/resolve/main/data/musan_music/train-00000.parquet",
        "local": "musan_hf/music_by3/train.parquet",
        "homepage": "https://www.openslr.org/17/",
        "mirror": "https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-music-cc-by-3.0",
        "license": "CC-BY-3.0 (per-recording; MUSAN music, documented instrumental)",
    },
    "musan_music_cc_by_sa_3": {
        "url": f"{_AKIKI}-musan-music-cc-by-sa-3.0/resolve/main/data/musan_music/train-00000.parquet",  # noqa: E501
        "local": "musan_hf/music_bysa3/train.parquet",
        "homepage": "https://www.openslr.org/17/",
        "mirror": "https://huggingface.co/datasets/Akiki-AI/akiki-vad-silero623-v1-musan-music-cc-by-sa-3.0",
        "license": "CC-BY-SA-3.0 (per-recording; MUSAN music, documented instrumental)",
    },
    "audioset_ontology": {
        "url": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
        "local": "audioset/ontology.json",
        "homepage": "https://github.com/audioset/ontology",
        "license": "CC-BY-SA-4.0",
        "note": "Used only to build the music / noise / speech class mapping.",
    },
}


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (atomic via .part)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "paribus-spike/0.0"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.rename(dest)
    return dest


def fetch(name: str) -> Path:
    """Download a dataset entry from :data:`DATASETS` into the raw cache."""
    entry = DATASETS[name]
    return download(entry["url"], RAW / entry["local"])


def load_manifest() -> list[dict[str, Any]]:
    with MANIFEST.open() as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    # SPIKE_SMOKE=N keeps every N-th clip: a quick end-to-end check, not a benchmark.
    step = int(os.environ.get("SPIKE_SMOKE", "1"))
    return entries[::step]


_PUNCT = re.compile(r"[^\w\s']|_", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """WER normalization: NFKC, lowercase, strip punctuation, squeeze spaces, Russian yo->ye."""
    text = (
        unicodedata.normalize("NFKC", text)
        .lower()
        .replace("\u2019", "'")
        .replace("\u0451", "\u0435")
    )
    text = _PUNCT.sub(" ", text)
    text = text.replace(" '", " ").replace("' ", " ")
    return " ".join(text.split())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float) + "\n")
