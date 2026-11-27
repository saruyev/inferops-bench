from pathlib import Path

import pytest
from pydantic import ValidationError

from paribus.engine.types import SampleRecord, TextSegment, Transcript, Word


def test_minimal_record_needs_only_id_and_audio() -> None:
    record = SampleRecord(id="a1", audio=Path("a1.wav"))

    assert record.channel is None
    assert record.output is None
    assert record.meta == {}


def test_record_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        SampleRecord(id="", audio=Path("a.wav"))


def test_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SampleRecord.model_validate({"id": "a", "audio": "a.wav", "lang": "en"})


def test_record_is_immutable() -> None:
    record = SampleRecord(id="a", audio=Path("a.wav"))

    with pytest.raises(ValidationError):
        record.id = "b"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("given", "expected"), [("en", "en"), (" RU ", "ru"), ("iw", "he"), ("jw", "jv")]
)
def test_language_is_normalized(given: str, expected: str) -> None:
    assert SampleRecord(id="a", audio=Path("a.wav"), language=given).language == expected


@pytest.mark.parametrize("given", ["eng", "en-US", "1x", ""])
def test_language_rejects_non_iso_639_1(given: str) -> None:
    with pytest.raises(ValidationError):
        SampleRecord(id="a", audio=Path("a.wav"), language=given)


def test_negative_channel_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SampleRecord(id="a", audio=Path("a.wav"), channel=-1)


@pytest.mark.parametrize("span", [(-0.1, 1.0), (2.0, 1.0)])
def test_bad_speech_segments_are_rejected(span: tuple[float, float]) -> None:
    with pytest.raises(ValidationError):
        SampleRecord(id="a", audio=Path("a.wav"), speech_segments=[span])


@pytest.mark.parametrize(("start", "end"), [(-0.1, 1.0), (2.0, 1.0)])
def test_word_and_segment_reject_bad_spans(start: float, end: float) -> None:
    with pytest.raises(ValidationError):
        Word(text="hi", start=start, end=end)
    with pytest.raises(ValidationError):
        TextSegment(start=start, end=end, text="hi")


def test_full_record_round_trips_through_json() -> None:
    transcript = Transcript(
        text="hello world",
        segments=[
            TextSegment(
                start=0.0,
                end=1.2,
                text="hello world",
                channel=1,
                words=[
                    Word(text="hello", start=0.0, end=0.5, signals={"logprob": -0.1}),
                    Word(text="world", start=0.6, end=1.2),
                ],
                signals={"avg_logprob": -0.2, "no_speech_prob": 0.01},
            )
        ],
    )
    record = SampleRecord(
        id="call-7/ch1",
        audio=Path("calls/7.webm"),
        channel=1,
        group_id="call-7",
        language="en",
        reference="hello world",
        output=transcript,
        speech_segments=[(0.0, 1.2)],
        meta={"tier": "gold", "agent_count": 2, "score": 0.5, "vip": True},
    )

    restored = SampleRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.meta["vip"] is True
    assert restored.meta["agent_count"] == 2
