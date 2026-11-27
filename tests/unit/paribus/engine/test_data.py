"""Tests for paribus.engine.data."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from paribus.engine.data import (
    MAX_ISSUES,
    FolderConfig,
    FolderSource,
    ManifestConfig,
    ManifestSource,
    SourceError,
)


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


def _source(path: Path, **settings: Any) -> ManifestSource:
    """A manifest source for `path` with license `private` unless overridden."""
    return ManifestSource(ManifestConfig(path=path, **{"license": "private", **settings}))


def _issues(error: pytest.ExceptionInfo[SourceError]) -> list[tuple[int, str]]:
    """The (line, message) pairs of a SourceError on a manifest, for compact assertions."""
    return [(int(issue.where.rsplit(":", 1)[1]), issue.message) for issue in error.value.issues]


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    """Only .jsonl and .csv are manifests."""
    with pytest.raises(ValueError, match=r"\.jsonl or \.csv"):
        _source(tmp_path / "data.json")


def test_name_defaults_to_file_stem(tmp_path: Path) -> None:
    """Without an explicit name the source is named after its file."""
    assert _source(tmp_path / "calls.jsonl").name == "calls"
    assert _source(tmp_path / "calls.jsonl", name="prod").name == "prod"


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

    records = list(_source(manifest, license="CC-BY-4.0").records())

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

    (record,) = _source(manifest).records()

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

    (record,) = _source(manifest, time_unit="ms").records()

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

    first, second = _source(manifest).records()

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
    source = _source(manifest, meta_types={"agents": "int", "score": "float", "vip": "bool"})

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
    source = _source(manifest, meta_types={"k": meta_type})

    with pytest.raises(SourceError) as error:
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
    source = _source(manifest, meta_types={"score": "float", "vip": "bool"})

    with pytest.raises(SourceError) as error:
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

    with pytest.raises(SourceError) as error:
        list(_source(manifest).records())

    lines = [line for line, _ in _issues(error)]
    messages = [message for _, message in _issues(error)]
    assert lines == [2, 3, 4, 5, 6]
    assert messages[0].startswith("not valid JSON")
    assert messages[1].startswith("lang: Extra inputs are not permitted")
    assert messages[2] == "duplicate id '1', first on line 1"
    assert messages[3].startswith("audio file not found")
    assert messages[4] == "each line must be a JSON object"
    assert not error.value.truncated
    assert f"{manifest}:2: not valid JSON" in str(error.value)


def test_issue_list_is_capped(tmp_path: Path) -> None:
    """A badly broken file reports the first MAX_ISSUES problems and says it stopped."""
    manifest = _jsonl(tmp_path / "m.jsonl", *["{bad"] * (MAX_ISSUES + 5))

    with pytest.raises(SourceError) as error:
        list(_source(manifest).records())

    assert len(error.value.issues) == MAX_ISSUES
    assert error.value.truncated
    assert "stopped after" in str(error.value)


def test_nothing_is_yielded_from_an_invalid_manifest(tmp_path: Path) -> None:
    """Validation covers the whole file before the first record comes out."""
    _audio(tmp_path, "1.wav")
    manifest = _jsonl(tmp_path / "m.jsonl", {"id": "1", "audio": "1.wav"}, "{bad")
    records = _source(manifest).records()

    with pytest.raises(SourceError):
        next(records)


def test_csv_extra_cells_are_reported(tmp_path: Path) -> None:
    """A row with more cells than the header is an error, not silently truncated."""
    _audio(tmp_path, "1.wav")
    manifest = tmp_path / "m.csv"
    manifest.write_text("id,audio\n1,1.wav,surprise\n", encoding="utf-8")

    with pytest.raises(SourceError) as error:
        list(_source(manifest).records())

    assert "Extra inputs are not permitted" in _issues(error)[0][1]


def test_config_rejects_unknown_keys_and_empty_license(tmp_path: Path) -> None:
    """Config typos fail like manifest typos, and a license must be declared."""
    with pytest.raises(ValidationError):
        ManifestConfig.model_validate({"path": "m.jsonl", "license": "x", "time_units": "ms"})
    with pytest.raises(ValidationError):
        ManifestConfig(path=tmp_path / "m.jsonl", license="")


def _folder(root: Path, **settings: Any) -> FolderSource:
    """A folder source for `root` with license `private` unless overridden."""
    return FolderSource(FolderConfig(root=root, **{"license": "private", **settings}))


def test_folder_ids_are_relative_paths_in_sorted_order(tmp_path: Path) -> None:
    """Ids are paths below the root without suffix; order is by path; case is ignored."""
    _audio(tmp_path, "b/2.flac", "a/1.WAV", "c.mp3", "notes.pdf")

    records = list(_folder(tmp_path, language="iw").records())

    assert [r.id for r in records] == ["a/1", "b/2", "c"]
    assert records[0].audio == (tmp_path / "a/1.WAV").resolve()
    assert {r.language for r in records} == {"he"}


def test_folder_skips_hidden_files_and_folders(tmp_path: Path) -> None:
    """Hidden files (incl. the `._*` files macOS leaves) and hidden folders are skipped."""
    _audio(tmp_path, "1.wav", "._1.wav", ".cache/2.wav")

    assert [r.id for r in _folder(tmp_path).records()] == ["1"]


def test_folder_does_not_follow_symlinked_folders(tmp_path: Path) -> None:
    """A symlink to another folder is not searched, so the walk cannot loop or escape."""
    _audio(tmp_path, "root/1.wav", "outside/2.wav")
    (tmp_path / "root/link").symlink_to(tmp_path / "outside", target_is_directory=True)

    assert [r.id for r in _folder(tmp_path / "root").records()] == ["1"]


def test_folder_name_defaults_to_folder_name(tmp_path: Path) -> None:
    """Without an explicit name the source is named after its folder."""
    assert _folder(tmp_path / "calls").name == "calls"


@pytest.mark.parametrize(("role", "field"), [("reference", "reference"), ("output", "output")])
def test_folder_transcripts_take_the_declared_role(tmp_path: Path, role: str, field: str) -> None:
    """A text file next to the audio becomes the gold reference or the pipeline output."""
    _audio(tmp_path, "1.wav", "2.wav")
    (tmp_path / "1.txt").write_text("  hello world\n", encoding="utf-8")

    first, second = _folder(tmp_path, transcripts=role).records()

    value = first.reference if field == "reference" else (first.output and first.output.text)
    assert value == "hello world"
    assert second.reference is None and second.output is None


def test_folder_ignores_transcripts_by_default(tmp_path: Path) -> None:
    """Without a declared role, text files next to audio are not read."""
    _audio(tmp_path, "1.wav")
    (tmp_path / "1.txt").write_text("hello", encoding="utf-8")

    (record,) = _folder(tmp_path).records()

    assert record.reference is None and record.output is None


def test_folder_reports_ids_that_collide(tmp_path: Path) -> None:
    """Two files that differ only in suffix would share an id; that is an error."""
    _audio(tmp_path, "a.mp3", "a.wav")

    with pytest.raises(SourceError) as error:
        list(_folder(tmp_path).records())

    (issue,) = error.value.issues
    assert issue.where == str(tmp_path / "a.wav")
    assert issue.message == f"same id 'a' as {tmp_path / 'a.mp3'}"


def test_folder_reports_a_transcript_that_is_not_utf8(tmp_path: Path) -> None:
    """A transcript in another encoding fails with its file, instead of turning into junk."""
    _audio(tmp_path, "1.wav")
    (tmp_path / "1.txt").write_bytes("привет".encode("cp1251"))

    with pytest.raises(SourceError) as error:
        list(_folder(tmp_path, transcripts="reference").records())

    assert error.value.issues[0].message == "transcript 1.txt is not UTF-8 text"


@pytest.mark.parametrize(
    ("make", "message"),
    [(False, "folder not found"), (True, "no audio files match")],
)
def test_folder_without_audio_is_an_error(tmp_path: Path, make: bool, message: str) -> None:
    """A missing or audio-less folder is almost always a wrong path or pattern."""
    root = tmp_path / "data"
    if make:
        root.mkdir()
        (root / "readme.txt").touch()

    with pytest.raises(SourceError) as error:
        list(_folder(root).records())

    assert error.value.issues[0].message.startswith(message)


def test_folder_config_validates_language_once(tmp_path: Path) -> None:
    """A bad folder language fails in the config, not once per file."""
    with pytest.raises(ValidationError):
        FolderConfig(root=tmp_path, license="private", language="english")
