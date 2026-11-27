"""Tests for paribus.engine.data."""

import json
from pathlib import Path

import pytest

from paribus.engine.data import MAX_ISSUES, ManifestError, ManifestSource


def _audio(folder: Path, *names: str) -> None:
    """Create empty placeholder audio files; only their existence is checked here."""
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _jsonl(path: Path, *rows: dict[str, object] | str) -> Path:
    """Write rows as a JSONL manifest; a str row is written verbatim (for broken lines)."""
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _issues(error: pytest.ExceptionInfo[ManifestError]) -> list[tuple[int, str]]:
    """The (line, message) pairs of a ManifestError, for compact assertions."""
    return [(issue.line, issue.message) for issue in error.value.issues]


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    """Only .jsonl and .csv are manifests."""
    with pytest.raises(ValueError, match=r"\.jsonl or \.csv"):
        ManifestSource(tmp_path / "data.json", license="private")


def test_name_defaults_to_file_stem(tmp_path: Path) -> None:
    """Without an explicit name the source is named after its file."""
    assert ManifestSource(tmp_path / "calls.jsonl", license="private").name == "calls"


def test_jsonl_records_come_out_in_order_with_resolved_paths(tmp_path: Path) -> None:
    """Relative audio paths resolve against the manifest folder; absolute ones are kept."""
    _audio(tmp_path, "a/1.wav")
    absolute = tmp_path / "elsewhere.wav"
    absolute.touch()
    manifest = _jsonl(
        tmp_path / "m.jsonl",
        {"id": "1", "audio": "a/1.wav", "language": "en"},
        "",
        {"id": "2", "audio": str(absolute)},
    )

    records = list(ManifestSource(manifest, license="CC-BY-4.0").records())

    assert [r.id for r in records] == ["1", "2"]
    assert records[0].audio == (tmp_path / "a/1.wav").resolve()
    assert records[1].audio == absolute


def test_jsonl_keeps_a_nested_transcript(tmp_path: Path) -> None:
    """JSONL can carry the pipeline's full output with segments and words."""
    _audio(tmp_path, "1.wav")
    output = {
        "text": "hi",
        "segments": [
            {
                "start": 0.0,
                "end": 0.4,
                "text": "hi",
                "words": [{"text": "hi", "start": 0.0, "end": 0.4}],
            }
        ],
    }
    manifest = _jsonl(tmp_path / "m.jsonl", {"id": "1", "audio": "1.wav", "output": output})

    (record,) = ManifestSource(manifest, license="private").records()

    assert record.output is not None
    assert record.output.segments is not None
    assert record.output.segments[0].words is not None
    assert record.output.segments[0].words[0].end == 0.4


def test_milliseconds_are_converted_to_seconds(tmp_path: Path) -> None:
    """With time_unit=ms every time field comes out in seconds."""
    _audio(tmp_path, "1.wav")
    row = {
        "id": "1",
        "audio": "1.wav",
        "group_id": "call",
        "offset": 30000,
        "speech_segments": [[0, 1500]],
        "output": {
            "text": "hi",
            "segments": [
                {
                    "start": 250,
                    "end": 750,
                    "text": "hi",
                    "words": [{"text": "hi", "start": 250, "end": 700}],
                }
            ],
        },
    }
    manifest = _jsonl(tmp_path / "m.jsonl", row)

    (record,) = ManifestSource(manifest, license="private", time_unit="ms").records()

    assert record.offset == 30.0
    assert record.speech_segments == [(0.0, 1.5)]
    assert record.output is not None and record.output.segments is not None
    segment = record.output.segments[0]
    assert (segment.start, segment.end) == (0.25, 0.75)
    assert segment.words is not None and segment.words[0].end == pytest.approx(0.7)


def test_csv_maps_flat_columns_meta_and_output_text(tmp_path: Path) -> None:
    """CSV gathers meta.<key> columns, reads output_text, and treats empty cells as unset."""
    _audio(tmp_path, "1.wav", "2.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "id,audio,channel,language,output_text,meta.tier,meta.agents\n"
        "1,1.wav,0,en,hello,gold,2\n"
        "2,2.wav,,,,,\n",
        encoding="utf-8",
    )

    first, second = ManifestSource(manifest, license="private").records()

    assert first.channel == 0
    assert first.output is not None and first.output.text == "hello"
    assert first.meta == {"tier": "gold", "agents": "2"}
    assert second.channel is None and second.language is None and second.meta == {}


def test_csv_meta_types_are_parsed(tmp_path: Path) -> None:
    """Declared metadata types turn CSV text into numbers and booleans."""
    _audio(tmp_path, "1.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "id,audio,meta.agents,meta.score,meta.vip,meta.note\n1,1.wav,2,0.5,TRUE,7\n",
        encoding="utf-8",
    )
    source = ManifestSource(
        manifest,
        license="private",
        meta_types={"agents": "int", "score": "float", "vip": "bool"},
    )

    (record,) = source.records()

    assert record.meta == {"agents": 2, "score": 0.5, "vip": True, "note": "7"}


@pytest.mark.parametrize(("meta_type", "value"), [("int", "two"), ("float", "x"), ("bool", "1")])
def test_csv_meta_value_that_does_not_parse_is_an_error(
    tmp_path: Path, meta_type: str, value: str
) -> None:
    """A value that does not match its declared type fails with its line, not silently."""
    _audio(tmp_path, "1.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"id,audio,meta.k\n1,1.wav,{value}\n", encoding="utf-8")
    source = ManifestSource(manifest, license="private", meta_types={"k": meta_type})  # type: ignore[dict-item]

    with pytest.raises(ManifestError) as error:
        list(source.records())

    assert _issues(error) == [(2, f"meta.k: {value!r} is not a valid {meta_type}")]


def test_jsonl_meta_types_are_checked(tmp_path: Path) -> None:
    """In JSONL declared types are checked; an int is accepted where a float is declared."""
    _audio(tmp_path, "1.wav", "2.wav")
    manifest = _jsonl(
        tmp_path / "m.jsonl",
        {"id": "1", "audio": "1.wav", "meta": {"score": 1, "vip": False}},
        {"id": "2", "audio": "2.wav", "meta": {"score": 1, "vip": 0}},
    )
    source = ManifestSource(
        manifest, license="private", meta_types={"score": "float", "vip": "bool"}
    )

    with pytest.raises(ManifestError) as error:
        list(source.records())

    assert _issues(error) == [(2, "meta.vip: 0 is not a valid bool")]


def test_problems_are_collected_with_line_numbers(tmp_path: Path) -> None:
    """Bad JSON, schema errors, duplicate ids and missing audio are all reported at once."""
    _audio(tmp_path, "1.wav")
    manifest = _jsonl(
        tmp_path / "m.jsonl",
        {"id": "1", "audio": "1.wav"},
        "{not json",
        {"id": "2", "audio": "1.wav", "lang": "en"},
        {"id": "1", "audio": "1.wav"},
        {"id": "3", "audio": "missing.wav"},
        "[1, 2]",
    )

    with pytest.raises(ManifestError) as error:
        list(ManifestSource(manifest, license="private").records())

    lines = [line for line, _ in _issues(error)]
    messages = [message for _, message in _issues(error)]
    assert lines == [2, 3, 4, 5, 6]
    assert messages[0].startswith("not valid JSON")
    assert messages[1].startswith("lang: Extra inputs are not permitted")
    assert messages[2] == "duplicate id '1', first on line 1"
    assert messages[3].startswith("audio file not found")
    assert messages[4] == "each line must be a JSON object"
    assert not error.value.truncated
    assert f"{manifest}:2:" in str(error.value)


def test_issue_list_is_capped(tmp_path: Path) -> None:
    """A badly broken file reports the first MAX_ISSUES problems and says it stopped."""
    manifest = _jsonl(tmp_path / "m.jsonl", *["{bad"] * (MAX_ISSUES + 5))

    with pytest.raises(ManifestError) as error:
        list(ManifestSource(manifest, license="private").records())

    assert len(error.value.issues) == MAX_ISSUES
    assert error.value.truncated
    assert "stopped after" in str(error.value)


def test_nothing_is_yielded_from_an_invalid_manifest(tmp_path: Path) -> None:
    """Validation covers the whole file before the first record comes out."""
    _audio(tmp_path, "1.wav")
    manifest = _jsonl(tmp_path / "m.jsonl", {"id": "1", "audio": "1.wav"}, "{bad")
    records = ManifestSource(manifest, license="private").records()

    with pytest.raises(ManifestError):
        next(records)


def test_csv_extra_cells_are_reported(tmp_path: Path) -> None:
    """A row with more cells than the header is an error, not silently truncated."""
    _audio(tmp_path, "1.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("id,audio\n1,1.wav,surprise\n", encoding="utf-8")

    with pytest.raises(ManifestError) as error:
        list(ManifestSource(manifest, license="private").records())

    assert "Extra inputs are not permitted" in _issues(error)[0][1]
