"""Tests for paribus.engine.audio, on small files generated per test."""

import wave
from pathlib import Path

import av
import numpy as np
import pytest

from paribus.engine.audio import AudioData, audio_hash, load_audio
from paribus.engine.types import SampleRecord

_RATE = 16000


def _tones(seconds: float = 1.0) -> np.ndarray:
    """Two channels: a 440 Hz tone at full scale and a 1 kHz tone at half scale."""
    t = np.arange(int(_RATE * seconds)) / _RATE
    return np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 1000 * t)])


def _wav(path: Path, samples: np.ndarray) -> Path:
    """Write (channels, frames) floats as a 16-bit wav with the standard library."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(samples.shape[0])
        out.setsampwidth(2)
        out.setframerate(_RATE)
        out.writeframes((samples.T * 32767).astype("<i2").tobytes())
    return path


def _m4a(path: Path, samples: np.ndarray) -> Path:
    """Encode (channels, frames) floats as stereo AAC in an m4a container with PyAV."""
    with av.open(str(path), "w") as out:
        stream = out.add_stream("aac", rate=_RATE, layout="stereo")
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.float32), format="fltp", layout="stereo"
        )
        frame.sample_rate = _RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return path


def _record(path: Path, channel: int | None = None) -> SampleRecord:
    """A record for one audio file."""
    return SampleRecord(id=path.stem, audio=path, channel=channel)


def _peak_hz(signal: np.ndarray, rate: int) -> float:
    """The frequency with the most energy in `signal`."""
    spectrum = np.abs(np.fft.rfft(signal))
    return float(np.fft.rfftfreq(signal.size, 1 / rate)[spectrum.argmax()])


def test_wav_decodes_to_float_channels_at_native_rate(tmp_path: Path) -> None:
    """All channels come back as float32 in [-1, 1], with duration from the frames."""
    audio = load_audio(_record(_wav(tmp_path / "a.wav", _tones())))

    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (2, _RATE)
    assert audio.sample_rate == _RATE
    assert audio.duration == 1.0
    assert np.abs(audio.samples).max() == pytest.approx(1.0, abs=1e-3)


def test_m4a_decodes_through_the_same_path(tmp_path: Path) -> None:
    """Compressed formats go through PyAV too; the tones survive encoding."""
    audio = load_audio(_record(_m4a(tmp_path / "a.m4a", _tones())))

    assert audio.samples.shape[0] == 2
    assert _peak_hz(audio.samples[0], audio.sample_rate) == pytest.approx(440, abs=2)
    assert _peak_hz(audio.samples[1], audio.sample_rate) == pytest.approx(1000, abs=2)


def test_channel_keeps_only_that_channel(tmp_path: Path) -> None:
    """A record of one channel gets that channel alone, shaped (1, frames)."""
    audio = load_audio(_record(_wav(tmp_path / "a.wav", _tones()), channel=1))

    assert audio.samples.shape == (1, _RATE)
    assert np.abs(audio.samples).max() == pytest.approx(0.5, abs=1e-3)


def test_channel_beyond_the_file_is_an_error(tmp_path: Path) -> None:
    """Asking for a channel the file does not have says how many it has."""
    path = _wav(tmp_path / "a.wav", _tones())

    with pytest.raises(ValueError, match=r"channel 2 requested, .* has 2"):
        load_audio(_record(path, channel=2))


def test_resampling_keeps_length_and_pitch(tmp_path: Path) -> None:
    """Resampling to 8 kHz halves the frames and leaves the 440 Hz tone where it was."""
    path = _wav(tmp_path / "a.wav", _tones())

    audio = load_audio(_record(path), sample_rate=8000)

    assert audio.sample_rate == 8000
    assert audio.samples.shape == (2, 8000)
    assert audio.samples.dtype == np.float32
    assert _peak_hz(audio.samples[0], 8000) == pytest.approx(440, abs=2)


def test_file_is_not_modified(tmp_path: Path) -> None:
    """Loading and resampling never write to the user's file."""
    path = _wav(tmp_path / "a.wav", _tones())
    before = path.read_bytes()

    load_audio(_record(path), sample_rate=8000)

    assert path.read_bytes() == before


def test_mono_averages_channels() -> None:
    """`mono()` is the mean of the channels, one-dimensional."""
    audio = AudioData(samples=np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), sample_rate=2)

    assert audio.mono().tolist() == [0.5, 0.5]


def test_broken_file_is_a_clear_error(tmp_path: Path) -> None:
    """A file that is not audio fails with its path, not with an ffmpeg error code."""
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not audio at all" * 10)

    with pytest.raises(ValueError, match=r"cannot decode .*bad\.wav"):
        load_audio(_record(path))


def test_hash_is_of_the_bytes_and_notices_edits(tmp_path: Path) -> None:
    """Same bytes give the same hash wherever they are; changed bytes give a new one."""
    first = _wav(tmp_path / "a.wav", _tones())
    copy = tmp_path / "copy.wav"
    copy.write_bytes(first.read_bytes())
    original = audio_hash(first)

    assert audio_hash(copy) == original
    _wav(first, _tones(seconds=0.5))
    assert audio_hash(first) != original
