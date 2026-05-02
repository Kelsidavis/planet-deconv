"""Build (frame, target) training pairs from one capture.

CLI:
    python pair_extract.py --ser /path/to/foo.ser --stack /path/to/foo.tif \\
        --out ./pairs/foo [--crop-size 256] [--max-shift 8] [--limit N]

Library entry:
    from pair_extract import extract_one
    stats = extract_one(video_path, stack_path, out_dir, ...)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from crop import crop_around, planet_centroid
from video import open_video


def laplacian_variance(arr: np.ndarray) -> float:
    f = arr.astype(np.float32)
    if f.ndim == 3:
        f = f.mean(axis=-1)
    lap = (
        f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4.0 * f[1:-1, 1:-1]
    )
    return float(lap.var())


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    mx = float(arr.max())
    if mx == 0:
        return arr.astype(np.uint8)
    return (arr.astype(np.float32) * (255.0 / mx)).clip(0, 255).astype(np.uint8)


def load_stack(path: Path, target_w: int, target_h: int) -> np.ndarray:
    """Load the AS!3 stack and resize it to the video frame dimensions.

    AS!3's drizzle factor (1.5x, 3.0x) makes the stack larger than the input;
    resizing back down brings them onto the same coordinate grid so the planet
    can be cropped from both at the same logical scale.
    """
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def phase_corr_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[int, int]:
    a = ref.astype(np.float32) - ref.mean()
    b = mov.astype(np.float32) - mov.mean()
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = A * np.conj(B)
    mag = np.abs(R) + 1e-10
    r = np.fft.irfft2(R / mag, s=a.shape)
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = a.shape
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return int(dy), int(dx)


def extract_one(
    video_path: Path,
    stack_path: Path,
    out_dir: Path,
    *,
    crop_size: int = 256,
    max_shift: int = 8,
    limit: int = 0,
    n_previews: int = 6,
    verbose: bool = True,
) -> dict:
    """Run the pair extraction for a single capture.

    Returns a stats dict {kept, rejected, total, video, stack, out}.
    Stops early and returns kept=0 if the capture is too small or unparsable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(exist_ok=True)

    vid = open_video(video_path)
    try:
        if crop_size > vid.width or crop_size > vid.height:
            if verbose:
                print(f"  skip {video_path.name}: crop {crop_size} > frame "
                      f"{vid.width}x{vid.height}")
            return {"kept": 0, "rejected": 0, "total": len(vid),
                    "video": str(video_path), "stack": str(stack_path),
                    "out": str(out_dir), "skipped": "frame-too-small"}

        n_total = len(vid) if limit <= 0 else min(limit, len(vid))
        stack = load_stack(stack_path, vid.width, vid.height)
        c = planet_centroid(stack)
        if c is None:
            if verbose:
                print(f"  skip {video_path.name}: no centroid in stack")
            return {"kept": 0, "rejected": 0, "total": n_total,
                    "video": str(video_path), "stack": str(stack_path),
                    "out": str(out_dir), "skipped": "no-stack-centroid"}
        target, _ = crop_around(stack, c[0], c[1], crop_size)

        inputs: list[np.ndarray] = []
        qualities: list[float] = []
        shifts: list[tuple[int, int]] = []
        indices: list[int] = []
        rejected = 0

        for i in range(n_total):
            try:
                frame = vid.read_frame(i)
            except Exception:
                rejected += 1
                continue
            # Always work in mono — RGB SER (e.g. WinJUPOS DeRot output)
            # would otherwise sneak a 3rd channel into phase correlation.
            if frame.ndim == 3:
                frame = frame[..., :3].mean(axis=-1).astype(np.uint8)
            c = planet_centroid(frame)
            if c is None:
                rejected += 1
                continue
            patch, _ = crop_around(frame, c[0], c[1], crop_size)
            dy, dx = phase_corr_shift(target, patch)
            if abs(dy) > max_shift or abs(dx) > max_shift:
                rejected += 1
                continue
            aligned = np.roll(patch, (dy, dx), axis=(0, 1))
            inputs.append(aligned)
            qualities.append(laplacian_variance(aligned))
            shifts.append((dy, dx))
            indices.append(i)

        if not inputs:
            return {"kept": 0, "rejected": rejected, "total": n_total,
                    "video": str(video_path), "stack": str(stack_path),
                    "out": str(out_dir), "skipped": "no-usable-frames"}

        inputs_arr = np.stack(inputs, axis=0)
        qualities_arr = np.array(qualities, dtype=np.float32)
        shifts_arr = np.array(shifts, dtype=np.int32)
        indices_arr = np.array(indices, dtype=np.int32)

        np.save(out_dir / "inputs.npy", inputs_arr)
        np.save(out_dir / "target.npy", target)
        np.savez(
            out_dir / "meta.npz",
            quality=qualities_arr,
            shifts=shifts_arr,
            frame_index=indices_arr,
            crop_size=np.int32(crop_size),
            video_path=np.array(str(video_path)),
            stack_path=np.array(str(stack_path)),
        )

        kept = len(inputs)
        if verbose:
            print(f"  kept {kept}/{n_total}  rej={rejected}  "
                  f"qmin={qualities_arr.min():.2f} qmax={qualities_arr.max():.2f}  "
                  f"size={inputs_arr.nbytes / 1e6:.1f} MB")

        if n_previews > 0:
            order = np.argsort(qualities_arr)
            picks = list(order[-(n_previews // 2):]) + list(order[: n_previews - n_previews // 2])
            for k, idx in enumerate(picks):
                inp = inputs_arr[idx]
                row = np.concatenate([inp, target], axis=1)
                tag = "best" if k < n_previews // 2 else "worst"
                Image.fromarray(row).save(
                    out_dir / "preview" / f"pair_{indices_arr[idx]:06d}_{tag}_q{qualities_arr[idx]:.2f}.png"
                )

        return {"kept": kept, "rejected": rejected, "total": n_total,
                "video": str(video_path), "stack": str(stack_path),
                "out": str(out_dir)}
    finally:
        vid.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ser", type=Path, required=True, help="input video (.ser or .avi)")
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--n-previews", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    extract_one(
        args.ser, args.stack, args.out,
        crop_size=args.crop_size, max_shift=args.max_shift,
        limit=args.limit, n_previews=args.n_previews,
    )


if __name__ == "__main__":
    main()
