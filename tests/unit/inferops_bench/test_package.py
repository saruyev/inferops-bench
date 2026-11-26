from importlib.metadata import version

import inferops_bench


def test_version_matches_installed_metadata() -> None:
    assert inferops_bench.__version__ == version("inferops-bench")
