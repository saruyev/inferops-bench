from importlib.metadata import version

import paribus


def test_version_matches_installed_metadata() -> None:
    assert paribus.__version__ == version("paribus")
