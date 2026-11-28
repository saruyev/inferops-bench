"""Builders of corrupted test sets from public speech, driven by YAML recipes.

This package lives outside `src/` on purpose: it is a tool for making benchmark data,
not part of the installed paribus. Its output is an ordinary paribus manifest, so the
test sets it builds are read like any user's data.
"""
