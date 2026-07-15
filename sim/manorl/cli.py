"""Shared strict parsing for ManoRL command-line booleans."""

from __future__ import annotations

import argparse


def parse_cli_bool(value: str) -> bool:
    """Parse the explicit true/false CLI contract used by ManoRL tools."""

    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")
