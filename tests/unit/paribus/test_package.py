"""Tests for the paribus package itself."""

from importlib.metadata import version

import paribus


def test_version_matches_installed_metadata() -> None:
    """The package version comes from the installed distribution metadata."""
    assert paribus.__version__ == version("paribus")
