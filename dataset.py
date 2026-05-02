"""PyTorch Datasets over (inputs.npy, target.npy, meta.npz) capture dirs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _random_scale(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                  scale_range: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same random zoom to (input, target, mask).

    Tensors are (C, H, W); scale chosen uniformly in `scale_range`. The
    scaled tensor is centered and padded with zeros (zoom-out) or
    center-cropped (zoom-in) back to the original H×W. Same scale for all
    three so they stay aligned.
    """
    lo, hi = scale_range
    if hi <= lo:
        return x, y, m
    s = float(torch.empty(1).uniform_(lo, hi).item())
    if abs(s - 1.0) < 1e-3:
        return x, y, m
    H, W = x.shape[-2:]
    new_h = max(8, int(round(H * s)))
    new_w = max(8, int(round(W * s)))

    def _resize(t, mode):
        return F.interpolate(t.unsqueeze(0), size=(new_h, new_w),
                             mode=mode, align_corners=False if mode == "bilinear" else None
                             ).squeeze(0)

    xs = _resize(x, "bilinear")
    ys = _resize(y, "bilinear")
    ms = _resize(m, "nearest")

    if new_h < H:
        py0 = (H - new_h) // 2
        py1 = H - new_h - py0
        px0 = (W - new_w) // 2
        px1 = W - new_w - px0
        xs = F.pad(xs, (px0, px1, py0, py1), value=0.0)
        ys = F.pad(ys, (px0, px1, py0, py1), value=0.0)
        ms = F.pad(ms, (px0, px1, py0, py1), value=0.0)
    elif new_h > H:
        cy = (new_h - H) // 2
        cx = (new_w - W) // 2
        xs = xs[..., cy:cy + H, cx:cx + W]
        ys = ys[..., cy:cy + H, cx:cx + W]
        ms = ms[..., cy:cy + H, cx:cx + W]
    return xs, ys, ms


def list_capture_dirs(root: str | Path) -> list[Path]:
    """Find all capture dirs under root that look like extracted pair sets."""
    root = Path(root)
    return sorted(p.parent for p in root.glob("*/inputs.npy"))


class PairDataset(Dataset):
    """One capture's worth of (frame -> stack) pairs.

    Single shared target per capture; per-frame input. Yields float32 tensors
    in [0, 1] with a leading channel dim.
    """

    def __init__(self, pair_dir: str | Path, indices: np.ndarray | None = None,
                 mask_threshold: float = 0.03, n_frames: int = 1):
        if n_frames < 1 or n_frames % 2 == 0:
            raise ValueError(f"n_frames must be odd and >= 1, got {n_frames}")
        pair_dir = Path(pair_dir)
        self.inputs = np.load(pair_dir / "inputs.npy", mmap_mode="r")
        self.target = np.load(pair_dir / "target.npy")
        meta = np.load(pair_dir / "meta.npz")
        self.quality = meta["quality"]
        self.frame_index = meta["frame_index"]
        self.n_frames = n_frames
        self.indices = (
            np.arange(len(self.inputs)) if indices is None else np.asarray(indices)
        )
        # Per-capture normalization: scale by the brightest pixel across
        # all input frames. We use input.max (not target.max) so the same
        # statistic is computable at inference time, where we have only
        # the input video. Without this train/infer symmetry, the model
        # learns one scaling and is fed another at deployment.
        denom = max(int(np.asarray(self.inputs).max()), 1)
        self._scale = 1.0 / float(denom)
        target_t = torch.from_numpy(self.target.astype(np.float32) * self._scale).clamp_(0, 1)
        self.target_t = target_t.unsqueeze(0)
        # mask_threshold is now relative to the per-capture normalized target,
        # so 0.03 means "3% of the brightest planet pixel".
        self.mask_t = (self.target_t > mask_threshold).float()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        if self.n_frames == 1:
            x = torch.from_numpy(self.inputs[j].astype(np.float32) * self._scale).clamp_(0, 1).unsqueeze(0)
        else:
            n = len(self.inputs)
            half = self.n_frames // 2
            # Take n_frames neighbors in the original extraction order.
            # Clip to capture bounds; if capture is smaller than n_frames we
            # repeat-pad the closest available frame.
            start = max(0, j - half)
            end = min(n, start + self.n_frames)
            start = max(0, end - self.n_frames)
            chunk = np.asarray(self.inputs[start:end], dtype=np.float32)
            if chunk.shape[0] < self.n_frames:
                pad = np.repeat(chunk[-1:], self.n_frames - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            x = torch.from_numpy(chunk * self._scale).clamp_(0, 1)
        return {
            "input": x,
            "target": self.target_t,
            "mask": self.mask_t,
            "quality": float(self.quality[j]),
            "frame_index": int(self.frame_index[j]),
        }


class MultiPairDataset(Dataset):
    """Concatenate multiple capture dirs into one Dataset.

    Yields the same dict shape as PairDataset plus a `capture_id` string.
    Each capture has its own target/mask; the flat index space covers all
    frames across all captures.
    """

    def __init__(self, capture_dirs: list[str | Path],
                 mask_threshold: float = 0.03, n_frames: int = 1):
        self.captures: list[PairDataset] = []
        self.capture_ids: list[str] = []
        self.n_frames = n_frames
        flat: list[tuple[int, int]] = []
        for cd in capture_dirs:
            cd = Path(cd)
            try:
                ds = PairDataset(cd, mask_threshold=mask_threshold, n_frames=n_frames)
            except FileNotFoundError:
                continue
            if len(ds) == 0:
                continue
            self.captures.append(ds)
            self.capture_ids.append(cd.name)
            for j in range(len(ds)):
                flat.append((len(self.captures) - 1, j))
        self.flat = flat

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, i: int) -> dict:
        cap_idx, frame_idx = self.flat[i]
        ds = self.captures[cap_idx]
        item = ds[frame_idx]
        item["capture_id"] = self.capture_ids[cap_idx]
        item["capture_idx"] = cap_idx
        return item

    def split_by_capture(self, val_capture_ids: list[str]) -> tuple["MultiPairDataset", "MultiPairDataset"]:
        """Return (train, val) where val holds out entire captures.

        This is a stricter generalization test than holding out frames within
        captures — the val set has poses, seeing distributions, and targets
        the model has never seen.
        """
        val_set = set(val_capture_ids)
        train = MultiPairDataset.__new__(MultiPairDataset)
        val = MultiPairDataset.__new__(MultiPairDataset)
        train.captures, val.captures = [], []
        train.capture_ids, val.capture_ids = [], []
        train.flat, val.flat = [], []
        for cap_idx, ds in enumerate(self.captures):
            cid = self.capture_ids[cap_idx]
            target = val if cid in val_set else train
            target.captures.append(ds)
            target.capture_ids.append(cid)
            new_idx = len(target.captures) - 1
            for j in range(len(ds)):
                target.flat.append((new_idx, j))
        return train, val

    def split_by_frame(self, val_frac: float, seed: int = 0) -> tuple["MultiPairDataset", "MultiPairDataset"]:
        """Return (train, val) sampled at random across captures.

        Same captures appear in both splits; just different frames. Easier
        signal — measures whether the model can fit data it's seen the
        target for, rather than generalize to unseen captures.
        """
        rng = np.random.default_rng(seed)
        n = len(self.flat)
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val_idx = set(int(i) for i in idx[:n_val])
        train = MultiPairDataset.__new__(MultiPairDataset)
        val = MultiPairDataset.__new__(MultiPairDataset)
        train.captures = self.captures
        val.captures = self.captures
        train.capture_ids = self.capture_ids
        val.capture_ids = self.capture_ids
        train.flat = [t for i, t in enumerate(self.flat) if i not in val_idx]
        val.flat = [t for i, t in enumerate(self.flat) if i in val_idx]
        return train, val
