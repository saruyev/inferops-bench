"""Build a test set from a recipe: phrases x conditions -> wav files + a paribus manifest.

Usage: `python -m testsets.build RECIPE --out DIR [--force]`

The set is written to `DIR/<recipe name>/`: `audio/<condition>/...wav`, `manifest.jsonl`
(read it with a `manifest` source) and `sources.json` (every input with its revision and
license). It is built in a temporary folder next to the target and moved into place at
the end, so an interrupted build never leaves a half-written set behind.

Every clip has its own random generator seeded from (recipe seed, condition, clip
number), so a rebuild gives byte-identical files and adding a condition changes no
existing clip.
"""

import argparse
import json
import shutil
import wave
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from paribus.engine.audio import load_audio
from paribus.engine.data import build_source
from paribus.engine.types import MetaValue, SampleRecord
from testsets import ops
from testsets.ops import Clip, Samples
from testsets.recipe import (
    Background,
    Codec,
    Condition,
    Gain,
    Generate,
    Mix,
    Op,
    Pad,
    Recipe,
    Resample,
    SpeechInput,
    load_recipe,
)


@dataclass(frozen=True)
class Phrase:
    """A source phrase, decoded once and corrupted by every speech condition.

    Attributes:
        key: `<source name>/<record id>`, unique across the recipe's inputs.
        record: The source record (language, reference, id).
        samples: Mono audio at the recipe's sample rate.
        speech: Ground-truth speech spans in seconds.
        truth: `alignment` if the spans come from the source, `utterance_bounds` if
            they are just the whole phrase.
    """

    key: str
    record: SampleRecord
    samples: Samples
    speech: list[ops.Span]
    truth: str


@dataclass(frozen=True)
class Pool:
    """Background clips of one recipe background, decoded at the recipe's sample rate."""

    clips: list[tuple[str, Samples]]
    licenses: Counter[str] = field(default_factory=Counter)


@dataclass
class _Usage:
    """What a build took from each input, for `sources.json`."""

    clips: Counter[str] = field(default_factory=Counter)


def build(recipe: Recipe, out_dir: Path, force: bool = False) -> Path:
    """Build the recipe's test set under `out_dir` and return its manifest path.

    Raises:
        FileExistsError: If the set already exists and `force` is not set.
        ValueError: If an input has fewer usable phrases than the recipe takes.
    """
    target = out_dir / recipe.name
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild it")
    partial = out_dir / f".{recipe.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    phrases = _load_phrases(recipe)
    pools = {
        name: _load_pool(background, recipe.sample_rate)
        for name, background in recipe.backgrounds.items()
    }
    usage = _Usage()
    records = [
        _write_clip(partial, condition, number, clip, meta, phrase)
        for condition in recipe.conditions
        for number, phrase, clip, meta in _clips(recipe, condition, phrases, pools, usage)
    ]
    _write_manifest(partial / "manifest.jsonl", records)
    _write_sources(partial / "sources.json", recipe, pools, usage)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    return target / "manifest.jsonl"


def _load_phrases(recipe: Recipe) -> list[Phrase]:
    """Decode every input's phrases and take `take` of each, chosen by the seed.

    Raises:
        ValueError: If two inputs share a source name, or an input has too few phrases
            within its duration filter.
    """
    phrases: list[Phrase] = []
    for number, speech in enumerate(recipe.speech):
        rng = np.random.default_rng([recipe.seed, number])
        candidates = _candidates(speech, recipe.sample_rate)
        if len(candidates) < speech.take:
            raise ValueError(
                f"speech input {number} has {len(candidates)} phrases within its duration "
                f"filter, the recipe takes {speech.take}"
            )
        chosen = sorted(rng.choice(len(candidates), size=speech.take, replace=False))
        phrases.extend(candidates[index] for index in chosen)
    keys = [phrase.key for phrase in phrases]
    if len(set(keys)) != len(keys):
        raise ValueError("two speech inputs share a source name; give them distinct `name`s")
    return phrases


def _candidates(speech: SpeechInput, sample_rate: int) -> list[Phrase]:
    """Decode an input's records and keep those within its duration filter."""
    source = build_source(speech.source)
    candidates = []
    for record in source.records():
        samples = load_audio(record, sample_rate).mono()
        duration = samples.shape[0] / sample_rate
        if speech.duration is not None and not speech.duration[0] <= duration <= speech.duration[1]:
            continue
        spans = record.speech_segments
        candidates.append(
            Phrase(
                key=f"{source.name}/{record.id}",
                record=record,
                samples=samples,
                speech=list(spans) if spans is not None else [(0.0, duration)],
                truth="alignment" if spans is not None else "utterance_bounds",
            )
        )
    return candidates


def _load_pool(background: Background, sample_rate: int) -> Pool:
    """Decode a background's clips, keeping only those whose license is allowed.

    Raises:
        ValueError: If no clip is left after the license filter.
    """
    source = build_source(background.source)
    clips: list[tuple[str, Samples]] = []
    licenses: Counter[str] = Counter()
    for record in source.records():
        license_name = (
            str(record.meta.get(background.license_column, ""))
            if background.license_column
            else source.license
        )
        if background.licenses is not None and license_name not in background.licenses:
            continue
        clips.append((f"{source.name}/{record.id}", load_audio(record, sample_rate).mono()))
        licenses[license_name] += 1
    if not clips:
        raise ValueError(f"background source {source.name!r} has no clip with an allowed license")
    return Pool(clips=clips, licenses=licenses)


def _clips(
    recipe: Recipe,
    condition: Condition,
    phrases: list[Phrase],
    pools: Mapping[str, Pool],
    usage: _Usage,
) -> list[tuple[int, Phrase | None, Clip, dict[str, MetaValue]]]:
    """Make every clip of one condition: one per phrase, or `count` without speech."""
    salt = zlib.crc32(condition.name.encode())
    if condition.speech == "none":
        assert condition.count is not None  # guaranteed by Condition's validator
        starts: list[tuple[Phrase | None, Clip | None]] = [(None, None)] * condition.count
    else:
        starts = [
            (phrase, Clip(phrase.samples, recipe.sample_rate, phrase.speech)) for phrase in phrases
        ]
    made = []
    for number, (phrase, clip) in enumerate(starts):
        rng = np.random.default_rng([recipe.seed, salt, number])
        meta: dict[str, MetaValue] = {"condition": condition.name}
        for op in condition.ops:
            clip = _apply(op, clip, rng, recipe.sample_rate, pools, meta, usage)
        assert clip is not None  # a no-speech condition starts with generate
        made.append((number, phrase, clip, meta))
    return made


def _apply(
    op: Op,
    clip: Clip | None,
    rng: np.random.Generator,
    sample_rate: int,
    pools: Mapping[str, Pool],
    meta: dict[str, MetaValue],
    usage: _Usage,
) -> Clip:
    """Apply one operation, drawing its ranges and recording what was drawn in `meta`."""
    if isinstance(op, Generate):
        seconds = ops.draw(op.seconds, rng)
        _put(meta, "seconds", round(seconds, 3))
        frames = int(seconds * sample_rate)
        return ops.generate(
            _fill(op.fill, frames, op.floor_dbfs, rng, pools, meta, usage), sample_rate
        )
    assert clip is not None  # only generate starts from nothing
    if isinstance(op, Mix):
        snr_db = ops.draw(op.snr_db, rng)
        _put(meta, "snr_db", round(snr_db, 2))
        background = _fill(op.background, clip.frames, 0.0, rng, pools, meta, usage)
        return ops.mix(clip, background, snr_db)
    if isinstance(op, Pad):
        before = int(ops.draw(op.before, rng) * clip.sample_rate)
        after = int(ops.draw(op.after, rng) * clip.sample_rate)
        _put(meta, "pad_before", round(before / clip.sample_rate, 3))
        _put(meta, "pad_after", round(after / clip.sample_rate, 3))
        filler = _fill(op.fill, before + after, op.floor_dbfs, rng, pools, meta, usage)
        return ops.pad(clip, filler[:before], filler[before:])
    if isinstance(op, Resample):
        _put(meta, "sample_rate_out", op.rate)
        return ops.resample(clip, op.rate)
    if isinstance(op, Codec):
        _put(meta, "codec", op.name if op.bitrate is None else f"{op.name}@{op.bitrate}")
        return ops.codec(clip, op.name, op.bitrate)
    if isinstance(op, Gain):
        db = ops.draw(op.db, rng)
        _put(meta, "gain_db", round(db, 2))
        return ops.gain(clip, db)
    raise TypeError(f"unhandled operation {op!r}")


def _fill(
    fill: str,
    frames: int,
    floor_dbfs: float,
    rng: np.random.Generator,
    pools: Mapping[str, Pool],
    meta: dict[str, MetaValue],
    usage: _Usage,
) -> Samples:
    """Make `frames` of a fill: digital zero, a noise floor, or a random background stretch."""
    if fill == "zero":
        return ops.zeros(frames)
    if fill == "floor":
        return ops.noise_floor(frames, floor_dbfs, rng)
    clips = pools[fill].clips
    background_id, samples = clips[int(rng.integers(len(clips)))]
    _put(meta, "background_id", background_id)
    usage.clips[background_id] += 1
    return ops.excerpt(samples, frames, rng) if frames else ops.zeros(0)


def _put(meta: dict[str, MetaValue], key: str, value: MetaValue) -> None:
    """Set `meta[key]`, or `key_2`, `key_3`... when an operation repeats in a condition."""
    name, number = key, 1
    while name in meta:
        number += 1
        name = f"{key}_{number}"
    meta[name] = value


def _write_clip(
    root: Path,
    condition: Condition,
    number: int,
    clip: Clip,
    meta: dict[str, MetaValue],
    phrase: Phrase | None,
) -> SampleRecord:
    """Write one clip as 16-bit wav and return its record, with the audio path relative."""
    name = phrase.key if phrase is not None else f"{number:04d}"
    relative = Path("audio") / condition.name / f"{name}.wav"
    _write_wav(root / relative, clip)
    if phrase is not None:
        meta = {**meta, "source_id": phrase.key, "speech_truth": phrase.truth}
    return SampleRecord(
        id=f"{condition.name}/{name}",
        audio=relative,
        language=phrase.record.language if phrase else None,
        reference=phrase.record.reference if phrase else None,
        speech_segments=[(round(start, 4), round(end, 4)) for start, end in clip.speech],
        meta=meta,
    )


def _write_wav(path: Path, clip: Clip) -> None:
    """Write a clip as mono 16-bit PCM, the format every model and tool reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(clip.samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(clip.sample_rate)
        out.writeframes(pcm.tobytes())


def _write_manifest(path: Path, records: list[SampleRecord]) -> None:
    """Write records as JSONL, one per line, with audio paths relative to the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            row = record.model_dump(mode="json", exclude_none=True)
            row["audio"] = record.audio.as_posix()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sources(path: Path, recipe: Recipe, pools: Mapping[str, Pool], usage: _Usage) -> None:
    """Write every input with its full config (dataset, revision) and license."""
    inputs: list[dict[str, Any]] = [
        {"role": "speech", "source": speech.source, "take": speech.take} for speech in recipe.speech
    ]
    for name, background in recipe.backgrounds.items():
        used = sum(count for clip, count in usage.clips.items() if clip in dict(pools[name].clips))
        inputs.append(
            {
                "role": f"background:{name}",
                "source": background.source,
                "clips_in_pool_by_license": dict(sorted(pools[name].licenses.items())),
                "clips_used": used,
            }
        )
    document = {"recipe": recipe.name, "seed": recipe.seed, "inputs": inputs}
    path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    """Command line: build one recipe."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("recipe", type=Path, help="recipe YAML file")
    parser.add_argument("--out", type=Path, required=True, help="folder to build the set in")
    parser.add_argument("--force", action="store_true", help="rebuild an existing set")
    args = parser.parse_args()
    manifest = build(load_recipe(args.recipe), args.out, force=args.force)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
