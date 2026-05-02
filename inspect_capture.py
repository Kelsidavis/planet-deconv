"""Inspect a SER planetary capture: header, metadata, per-frame quality, previews.

Usage:
    python inspect_capture.py /path/to/capture.ser [--out ./inspect_out]
        [--n-previews 8] [--stride 1] [--crop-size 256]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from crop import crop_around, planet_centroid
from ser import SerFile
from settings import parse_camera_settings


def laplacian_variance(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    f = frame.astype(np.float32)
    lap = (
        f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4.0 * f[1:-1, 1:-1]
    )
    return float(lap.var())


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    mx = int(arr.max())
    if mx == 0:
        return arr.astype(np.uint8)
    return (arr.astype(np.float32) * (255.0 / mx)).clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ser", type=Path)
    ap.add_argument("--out", type=Path, default=Path("./inspect_out"))
    ap.add_argument("--n-previews", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame step for the quality scan (>1 = faster, sparser)")
    ap.add_argument("--crop-size", type=int, default=256,
                    help="square crop side (px) around the planet centroid")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    settings_path = args.ser.with_suffix(".CameraSettings.txt")
    settings = parse_camera_settings(settings_path) if settings_path.exists() else {}

    with SerFile(args.ser) as ser:
        h = ser.header
        n = len(ser)
        print(f"SER  {args.ser}")
        print(f"  {h.width}x{h.height}  bit_depth={h.bit_depth}  color_id={h.color_id}  frames={n}")
        print(f"  observer={h.observer!r}  instrument={h.instrument!r}  telescope={h.telescope!r}")
        if settings:
            wanted = ["Exposure", "Gain", "Offset", "ZWO FilterWheel (1)",
                      "Temperature", "ActualFrameRate", "FrameCount",
                      "Duration", "OnStep Telescope"]
            for k in wanted:
                if k in settings:
                    print(f"  [settings] {k}: {settings[k]}")

        if args.crop_size > min(h.width, h.height):
            raise SystemExit(
                f"--crop-size {args.crop_size} > frame {h.width}x{h.height}"
            )

        idxs = list(range(0, n, args.stride))
        qualities = np.full(len(idxs), np.nan, dtype=np.float32)
        centers = np.full((len(idxs), 2), np.nan, dtype=np.float32)
        n_skipped = 0
        for i, frame_idx in enumerate(idxs):
            frame = ser.read_frame(frame_idx)
            c = planet_centroid(frame)
            if c is None:
                n_skipped += 1
                continue
            centers[i] = c
            patch, _ = crop_around(frame, c[0], c[1], args.crop_size)
            qualities[i] = laplacian_variance(patch)

        valid = ~np.isnan(qualities)
        if not valid.any():
            raise SystemExit("no frames with detectable signal")
        q_valid = qualities[valid]
        print(f"  centroid drift: y={np.nanstd(centers[:,0]):.1f}px  "
              f"x={np.nanstd(centers[:,1]):.1f}px  (skipped={n_skipped})")
        print(f"  quality (Laplacian var on {args.crop_size}px crop): "
              f"min={q_valid.min():.1f}  median={np.median(q_valid):.1f}  "
              f"max={q_valid.max():.1f}  std={q_valid.std():.1f}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(idxs, qualities, lw=0.6)
        axes[0].set_ylabel("Laplacian var (crop)")
        axes[0].set_title(f"{args.ser.name}  —  crop {args.crop_size}px, stride {args.stride}")
        axes[1].plot(idxs, centers[:, 1], lw=0.5, label="cx")
        axes[1].plot(idxs, centers[:, 0], lw=0.5, label="cy")
        axes[1].set_xlabel("frame index")
        axes[1].set_ylabel("centroid (px)")
        axes[1].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "quality.png", dpi=120)
        plt.close(fig)

        valid_idx = np.where(valid)[0]
        order_in_valid = valid_idx[np.argsort(q_valid)]
        best_pos = int(order_in_valid[-1])
        worst_pos = int(order_in_valid[0])
        best = idxs[best_pos]
        worst = idxs[worst_pos]
        n_mid = max(args.n_previews - 2, 0)
        if n_mid > 0:
            mid = [idxs[i] for i in np.linspace(0, len(idxs) - 1, n_mid, dtype=int)]
        else:
            mid = []
        picks = sorted(set([best, worst, *mid]))
        for frame_idx in picks:
            frame = ser.read_frame(frame_idx)
            tag = "best" if frame_idx == best else "worst" if frame_idx == worst else "mid"
            Image.fromarray(to_uint8(frame)).save(
                args.out / f"frame_{frame_idx:06d}_{tag}.png"
            )
            c = planet_centroid(frame)
            if c is not None:
                patch, _ = crop_around(frame, c[0], c[1], args.crop_size)
                Image.fromarray(to_uint8(patch)).save(
                    args.out / f"crop_{frame_idx:06d}_{tag}.png"
                )

        print(f"  wrote {len(picks)} previews + crops + quality.png to {args.out}")


if __name__ == "__main__":
    main()
