"""Shared data types: what a sample is and what a transcript looks like.

The same `Transcript` type holds a user's existing pipeline output and the output of
paribus's own ASR adapters, so the two can be compared without conversions.

All times are seconds (float) from the start of the sample's audio, after channel
selection. Other units are converted at the boundaries (import, report).
"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Raw model signals under their own names (e.g. "logprob", "avg_logprob", "no_speech_prob").
# Their level, scale and direction are declared by the ASR adapter profile, not normalized here.
type Signals = dict[str, float]
type MetaValue = str | int | float | bool

# Obsolete ISO 639-1 codes still used by some datasets and models.
_LEGACY_LANGUAGE_CODES = {"iw": "he", "ji": "yi", "in": "id", "jw": "jv"}


class _Frozen(BaseModel):
    """Base for all types here: immutable, and unknown fields are an error.

    Immutable, so records and transcripts can be cached by content hash safely.
    Unknown fields fail, so a typo in a manifest (`lang` for `language`) is reported
    instead of being silently dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _check_span(start: float, end: float) -> None:
    """Raise ValueError unless 0 <= start <= end.

    Containment (a word inside its segment, a segment inside the audio) is not checked:
    real models emit words that overrun their segment by tens of milliseconds, and such
    overruns are better measured as a feature than rejected as invalid data.
    """
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")


class Word(_Frozen):
    """One recognized word with its time span and the raw signals the model gave for it.

    `signals` keeps values under the model's own names (e.g. `logprob`, `prob`); there is
    no normalized confidence, because models do not share a scale.
    """

    text: str
    start: float
    end: float
    signals: Signals = Field(default_factory=dict)

    @model_validator(mode="after")
    def _span(self) -> Self:
        """Check that the word's time span is valid."""
        _check_span(self.start, self.end)
        return self


class TextSegment(_Frozen):
    """A stretch of recognized text, optionally with its words and per-segment signals.

    `channel` says which audio channel the segment came from when one record covers
    several channels. Words are nested here; a model that returns only words gets them
    wrapped into one segment by its adapter. Segments are not required to be sorted:
    in multichannel output, segments of different channels interleave.
    """

    start: float
    end: float
    text: str
    channel: int | None = Field(default=None, ge=0)
    words: list[Word] | None = None
    signals: Signals = Field(default_factory=dict)

    @model_validator(mode="after")
    def _span(self) -> Self:
        """Check that the segment's time span is valid."""
        _check_span(self.start, self.end)
        return self


class Transcript(_Frozen):
    """ASR output for one record: full text, optional segments, transcript-level signals."""

    text: str
    segments: list[TextSegment] | None = None
    signals: Signals = Field(default_factory=dict)


class SampleRecord(_Frozen):
    """One unit the user's pipeline sends to ASR, after the pipeline's own chunking.

    paribus takes records as they are and never re-chunks them on load. A record is a
    whole file, one channel of a file (`channel`), or a chunk the pipeline cut earlier.
    Records that come from the same call, conversation or batch share a `group_id`;
    `offset` is where this record starts on the group's timeline, in seconds. Times
    inside the record stay local: time in the group = `offset` + local time.

    Optional fields: `language` is the expected language (ISO 639-1), `reference` a gold
    transcript, `output` the pipeline's existing transcript (offline mode),
    `speech_segments` ground-truth speech spans (synthetic test sets only), and `meta`
    flat user metadata that slice search can use as features. Duration, sample rate and
    content hash are not stored; they are computed from the audio when it is loaded.
    """

    id: str = Field(min_length=1)
    audio: Path
    channel: int | None = Field(default=None, ge=0)
    group_id: str | None = None
    offset: float | None = Field(default=None, ge=0)
    language: str | None = None
    reference: str | None = None
    output: Transcript | None = None
    speech_segments: list[tuple[float, float]] | None = None
    meta: dict[str, MetaValue] = Field(default_factory=dict)

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        """Normalize to a lowercase ISO 639-1 code, mapping obsolete codes (iw -> he)."""
        if value is None:
            return None
        code = value.strip().lower()
        code = _LEGACY_LANGUAGE_CODES.get(code, code)
        if len(code) != 2 or not code.isalpha():
            raise ValueError(f"language must be an ISO 639-1 code like 'en', got {value!r}")
        return code

    @field_validator("speech_segments")
    @classmethod
    def _speech_segments(
        cls, value: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        """Check that every ground-truth speech span is valid."""
        for start, end in value or []:
            _check_span(start, end)
        return value

    @model_validator(mode="after")
    def _offset_needs_group(self) -> Self:
        """Reject an offset without a group: it is a position on the group's timeline."""
        if self.offset is not None and self.group_id is None:
            raise ValueError("offset is a position on a group's timeline and needs group_id")
        return self
