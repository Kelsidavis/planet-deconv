"""Run a trained model on a fresh capture (no target needed).

CLI:
    python infer.py --video PATH --ckpt PATH --out DIR
        [--indices "0,100,500,1000"] [--all] [--stride N]
        [--stack PATH]   # optional: include AS3 stack in the triptych

Inference normalization:
    Training divided by per-capture `target.max()`. At inference we don't
    have a target, so we approximate it with the max value found across a
    sample of input frames. This is usually slightly higher than the target
    max (raw frames have more noise spikes than the stacked target), so
    outputs may be a touch dim — denormalization with the same constant
    keeps shapes correct, and a final clip+rescale brings the display back.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from crop import crop_around, planet_centroid
from model import TinyUNet
from pair_extract import phase_corr_shift
from video import open_video


def to_uint8(arr: np.ndarray, hi: float | None = None) -> np.ndarray:
    """Convert any-shape float/int array to uint8 for saving.

    If `hi` is given, scale [0, hi] -> [0, 255]; otherwise scale by max.
    """
    if arr.dtype == np.uint8:
        return arr
    if hi is None:
        hi = float(arr.max()) if arr.size else 1.0
    if hi <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return (arr.astype(np.float32) * (255.0 / hi)).clip(0, 255).astype(np.uint8)


def soft_disk_mask(frame: np.ndarray, low: float = 0.05, high: float = 0.20) -> np.ndarray:
    """Build a soft 0..1 mask of the planet from a single normalized frame.

    Pixels below `low` are 0 (background, untouched), above `high` are 1
    (planet, model takes over), with a smooth transition in between. The
    smoothness avoids visible edges where the blend transitions.
    """
    f = frame.astype(np.float32)
    if f.ndim == 3:
        f = f.mean(axis=-1)
    m = np.clip((f - low) / max(high - low, 1e-6), 0.0, 1.0)
    # Smooth the mask edges so the blend doesn't show as a hard outline.
    # Cheap blur via 5x5 box (separable means).
    pad = np.pad(m, 2, mode="edge")
    out = np.zeros_like(m)
    for dy in range(5):
        for dx in range(5):
            out += pad[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out / 25.0


def estimate_capture_max(vid, n_samples: int = 24) -> int:
    """Find the brightest pixel across a sample of frames.

    Used as the per-capture normalization constant at inference, in place of
    the `target.max()` that training used.
    """
    n = len(vid)
    if n == 0:
        return 1
    sample_idx = np.linspace(0, n - 1, min(n_samples, n), dtype=int)
    mx = 0
    for i in sample_idx:
        f = vid.read_frame(int(i))
        if f.ndim == 3:
            f = f[..., :3].mean(axis=-1).astype(np.uint8)
        mx = max(mx, int(f.max()))
    return max(mx, 1)


def gather_burst(vid, center_idx: int, n_frames: int, crop_size: int) -> np.ndarray | None:
    """Return (n_frames, crop_size, crop_size) uint8 burst centered on `center_idx`.

    Each frame is cropped around its own planet centroid, then phase-correlated
    against the center frame so the burst is registered.
    """
    n = len(vid)
    half = n_frames // 2
    start = max(0, center_idx - half)
    end = min(n, start + n_frames)
    start = max(0, end - n_frames)

    # Read center first to anchor phase correlation
    center_frame = vid.read_frame(center_idx)
    if center_frame.ndim == 3:
        center_frame = center_frame[..., :3].mean(axis=-1).astype(np.uint8)
    c = planet_centroid(center_frame)
    if c is None:
        return None
    center_patch, _ = crop_around(center_frame, c[0], c[1], crop_size)

    burst = np.zeros((n_frames, crop_size, crop_size), dtype=np.uint8)
    burst_center = center_idx - start
    for off, idx in enumerate(range(start, end)):
        if idx == center_idx:
            burst[off] = center_patch
            continue
        try:
            frame = vid.read_frame(idx)
        except IOError:
            burst[off] = center_patch  # pad with center on read error
            continue
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1).astype(np.uint8)
        cc = planet_centroid(frame)
        if cc is None:
            burst[off] = center_patch
            continue
        patch, _ = crop_around(frame, cc[0], cc[1], crop_size)
        dy, dx = phase_corr_shift(center_patch, patch)
        burst[off] = np.roll(patch, (dy, dx), axis=(0, 1))

    # Pad if capture had fewer than n_frames
    real_len = end - start
    if real_len < n_frames:
        last = burst[real_len - 1] if real_len > 0 else center_patch
        for off in range(real_len, n_frames):
            burst[off] = last
    return burst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--indices", default=None,
                    help="comma-separated frame indices; default = 8 evenly spaced")
    ap.add_argument("--all", action="store_true", help="process every frame")
    ap.add_argument("--stride", type=int, default=1, help="step when --all")
    ap.add_argument("--stack", type=Path, default=None,
                    help="optional AS3 unsharpened TIF; appended to triptych")
    ap.add_argument("--blend", action="store_true",
                    help="blend model output with input via planet mask "
                         "(removes dark-halo / square-edge artifacts)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    n_frames = int(saved_args.get("n_frames", 1))
    base = int(saved_args.get("base", 32))
    print(f"checkpoint: n_frames={n_frames}  base={base}  "
          f"epoch={ckpt.get('epoch')}  val_psnr={ckpt.get('val_psnr'):.2f}")

    model = TinyUNet(in_ch=n_frames, base=base).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    vid = open_video(args.video)
    print(f"video: {args.video.name}  {vid.width}x{vid.height}  frames={len(vid)}")

    cap_max = estimate_capture_max(vid)
    scale = 1.0 / float(cap_max)
    print(f"estimated capture max: {cap_max}  scale=1/{cap_max:.3g}")

    stack_patch = None
    if args.stack is not None:
        from pair_extract import load_stack

        stack = load_stack(args.stack, vid.width, vid.height)
        c = planet_centroid(stack)
        if c is not None:
            stack_patch, _ = crop_around(stack, c[0], c[1], args.crop_size)

    if args.all:
        idxs = list(range(0, len(vid), args.stride))
    elif args.indices:
        idxs = [int(s) for s in args.indices.split(",") if s.strip()]
    else:
        idxs = list(np.linspace(0, len(vid) - 1, 8, dtype=int))
    print(f"processing {len(idxs)} frame(s)")

    saved = 0
    skipped = 0
    for i, fi in enumerate(idxs):
        burst = gather_burst(vid, int(fi), n_frames, args.crop_size)
        if burst is None:
            skipped += 1
            continue
        x = torch.from_numpy(burst.astype(np.float32) * scale).clamp_(0, 1)
        x = x.unsqueeze(0).to(device)  # (1, n_frames, H, W)
        with torch.no_grad():
            pred = model.predict(x)
        pred_np = pred[0, 0].cpu().numpy()  # already clamped to [0,1]

        center_burst = burst[n_frames // 2]
        if args.blend:
            center_norm = center_burst.astype(np.float32) * scale
            m = soft_disk_mask(center_norm)
            pred_np = pred_np * m + center_norm * (1.0 - m)
            pred_np = np.clip(pred_np, 0.0, 1.0)
        # Auto-stretch each panel by its own max so brightness scales match
        # visually. The model output already lives in [0,1]; the input is
        # raw uint8, the stack is uint8 — autostretch all three.
        in_disp = to_uint8(center_burst)
        out_disp = to_uint8(pred_np)
        parts = [in_disp, out_disp]
        if stack_patch is not None:
            parts.append(to_uint8(stack_patch))
        row = np.concatenate(parts, axis=1)
        Image.fromarray(row).save(args.out / f"frame_{int(fi):06d}.png")
        saved += 1

    vid.close()
    print(f"saved {saved} previews, skipped {skipped}, into {args.out}")


if __name__ == "__main__":
    main()
