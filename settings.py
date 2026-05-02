"""Parser for SharpCap-emitted CameraSettings.txt sidecar files."""
from __future__ import annotations

from pathlib import Path


def parse_camera_settings(path: str | Path) -> dict[str, str]:
    """Parse a SharpCap CameraSettings.txt into a flat key->value dict.

    Section headers like [QHY183M] and blank lines are skipped. Values are
    kept as raw strings — the caller chooses how to interpret them.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out
