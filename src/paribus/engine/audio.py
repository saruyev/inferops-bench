"""Audio loading: decode a record's file, pick its channel, resample in memory.

Files are never modified: the user's pipeline keeps getting the original, and every
consumer (VAD, tagger, ASR adapter) asks for the sample rate it needs. Decoding goes
through PyAV, whose wheels bundle ffmpeg, so wav, flac, mp3, ogg, opus, m4a and webm all
take one path with no system dependency. Resampling uses soxr, which is fast, high
quality and deterministic.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import soxr

from paribus.engine.types import SampleRecord

type Samples = npt.NDArray[np.float32]

# Files are hashed in blocks of this size, so a long recording is never read into memory.
_HASH_BLOCK = 1 << 20


@dataclass(frozen=True)
class AudioData:
    """Decoded audio of one record.

    Attributes:
        samples: float32 in [-1, 1], shaped (channels, frames). A record with `channel`
            set has exactly one channel here.
        sample_rate: Frames per second of `samples`.
    """

    samples: Samples
    sample_rate: int

    @property
    def duration(self) -> float:
        """Length in seconds, from the decoded frames rather than container metadata.

        Container metadata is unreliable for some formats (mp3 without a header, webm
        written by browsers), and slice features must not depend on it.
        """
        return float(self.samples.shape[1] / self.sample_rate)

    def mono(self) -> Samples:
        """Return the channels averaged into one, shaped (frames,).

        This is the consumer's choice, not the loader's: averaging the two sides of a
        phone call mixes two speakers, which some features want and others must avoid.
        """
        return self.samples.mean(axis=0, dtype=np.float32)


def load_audio(record: SampleRecord, sample_rate: int | None = None) -> AudioData:
    """Decode the record's audio, keep its channel if set, and resample if asked.

    Args:
        record: The record whose `audio` file is decoded.
        sample_rate: Rate the consumer needs; None keeps the file's own rate.

    Raises:
        ValueError: If the file cannot be decoded, has no audio stream, or has no
            channel with the record's `channel` number.
    """
    samples, native_rate = _decode(record.audio)
    if record.channel is not None:
        samples = _select_channel(samples, record.channel, record.audio)
    if sample_rate is not None and sample_rate != native_rate:
        samples = _resample(samples, native_rate, sample_rate)
        native_rate = sample_rate
    return AudioData(samples=samples, sample_rate=native_rate)


def audio_hash(path: Path) -> str:
    """Return the sha256 of the file's bytes, as hex.

    The hash is of the bytes, not of decoded samples: it is cheaper and does not change
    when the decoder is upgraded. Within one process it is remembered by path, size and
    modification time, so a file is read once however many consumers ask.
    """
    stat = path.stat()
    return _hash_file(path.resolve(), stat.st_size, stat.st_mtime_ns)


def _decode(path: Path) -> tuple[Samples, int]:
    """Decode the first audio stream of a file to float32 (channels, frames) at its rate.

    PyAV hands out frames in the codec's own sample format (int16 packed for wav, float
    planar for aac); a resampler at the native rate and layout only converts them to
    float planar, it does not change the rate.

    Raises:
        ValueError: If the file cannot be decoded or has no audio stream.
    """
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"no audio stream in {path}")
            stream = container.streams.audio[0]
            converter = av.AudioResampler(format="fltp", layout=stream.layout, rate=stream.rate)
            chunks = [
                converted.to_ndarray()
                for frame in container.decode(stream)
                for converted in converter.resample(frame)
            ]
            chunks += [converted.to_ndarray() for converted in converter.resample(None)]
            rate, channels = stream.rate, stream.codec_context.channels
    except av.FFmpegError as error:
        raise ValueError(f"cannot decode {path}: {error}") from None
    if not chunks:
        return np.zeros((channels, 0), dtype=np.float32), rate
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False), rate


def _select_channel(samples: Samples, channel: int, path: Path) -> Samples:
    """Keep one channel, still shaped (1, frames).

    Raises:
        ValueError: If the file has no channel with that number.
    """
    if channel >= samples.shape[0]:
        raise ValueError(f"channel {channel} requested, {path} has {samples.shape[0]}")
    return samples[channel : channel + 1]


def _resample(samples: Samples, from_rate: int, to_rate: int) -> Samples:
    """Resample every channel with soxr's high-quality setting."""
    # soxr works on (frames, channels); our layout is (channels, frames).
    resampled = soxr.resample(samples.T, from_rate, to_rate, quality="HQ")
    return np.ascontiguousarray(resampled.T, dtype=np.float32).reshape(samples.shape[0], -1)


@lru_cache(maxsize=4096)
def _hash_file(path: Path, size: int, mtime_ns: int) -> str:
    """Hash a file's bytes; size and mtime are in the cache key so an edit is noticed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()
