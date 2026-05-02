"""Locate the planet in a frame and produce a fixed-size crop around it.

Centroid via background-subtracted center-of-mass — robust to mount drift,
moderate noise, and a few hot pixels.
"""
from __future__ import annotations

import numpy as np


def planet_centroid(frame: np.ndarray, bg_quantile: float = 0.1) -> tuple[float, float] | None:
    """Return (cy, cx) of the planet, or None if the frame has no signal."""
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    f = frame.astype(np.float32)
    bg = float(np.quantile(f, bg_quantile))
    f = np.clip(f - bg, 0.0, None)
    total = float(f.sum())
    if total <= 0.0:
        return None
    h, w = f.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    cy = float((ys * f).sum() / total)
    cx = float((xs * f).sum() / total)
    return cy, cx


def crop_around(frame: np.ndarray, cy: float, cx: float, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a square `size`-by-`size` patch centered on (cy, cx).

    Returns (patch, (y0, x0)). The crop is clipped to frame bounds, so patches
    near edges shift inward to keep the requested size.
    """
    h, w = frame.shape[:2]
    if size > h or size > w:
        raise ValueError(f"crop size {size} larger than frame {h}x{w}")
    half = size // 2
    y0 = int(round(cy - half))
    x0 = int(round(cx - half))
    y0 = max(0, min(h - size, y0))
    x0 = max(0, min(w - size, x0))
    return frame[y0:y0 + size, x0:x0 + size], (y0, x0)
