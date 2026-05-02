"""Train TinyUNet across multiple capture dirs with held-out captures for val.

Usage:
    python train_multi.py --root /media/k/Expansion/deconv_pairs \
        --out ./runs/multi_001 [--val-captures NAME1 NAME2 ...] [--epochs 30]

If --val-captures is omitted, two captures are auto-selected as held-out.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import MultiPairDataset, _random_scale, list_capture_dirs
from model import TinyUNet


def random_scale_batch(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                       scale_range: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-sample random scale to a batch (different scale per sample)."""
    if scale_range is None:
        return x, y, m
    out_x, out_y, out_m = [], [], []
    for i in range(x.size(0)):
        xi, yi, mi = _random_scale(x[i], y[i], m[i], scale_range)
        out_x.append(xi); out_y.append(yi); out_m.append(mi)
    return torch.stack(out_x), torch.stack(out_y), torch.stack(out_m)


def charbonnier(pred, tgt, mask=None, eps=1e-3):
    err = torch.sqrt((pred - tgt) ** 2 + eps * eps)
    if mask is None:
        return err.mean()
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def fourier_l1(pred, tgt):
    P = torch.fft.rfft2(pred)
    T = torch.fft.rfft2(tgt)
    return (P.abs() - T.abs()).abs().mean()


def psnr(pred, tgt, mask=None) -> float:
    if mask is None:
        mse = ((pred - tgt) ** 2).mean().item()
    else:
        mse = (((pred - tgt) ** 2) * mask).sum().item() / max(float(mask.sum().item()), 1.0)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def to_uint8_img(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().clamp(0, 1).numpy()
    if a.ndim == 3:
        a = a[0]
    return (a * 255.0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="parent dir containing capture subdirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-mode", choices=["capture", "frame"], default="capture",
                    help="capture: hold out entire captures (true generalization); "
                         "frame: hold out random frames across captures (fit signal)")
    ap.add_argument("--val-captures", nargs="*", default=None,
                    help="capture dir names to hold out as val (capture mode only)")
    ap.add_argument("--n-val-captures", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of frames held out (frame mode only)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--fft-weight", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=1,
                    help="burst size (odd, >=1). 1 = single-frame baseline.")
    ap.add_argument("--scale-aug-min", type=float, default=1.0,
                    help="min scale factor for random zoom aug (1.0 = off)")
    ap.add_argument("--scale-aug-max", type=float, default=1.0,
                    help="max scale factor for random zoom aug (1.0 = off)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cap_dirs = list_capture_dirs(args.root)
    print(f"found {len(cap_dirs)} captures in {args.root}")
    if not cap_dirs:
        raise SystemExit("no captures with inputs.npy found")

    full = MultiPairDataset(cap_dirs, n_frames=args.n_frames)
    print(f"n_frames per sample: {args.n_frames}")
    print(f"total frames: {len(full)} across {len(full.captures)} captures")
    print("frames per capture:")
    for cid, ds in zip(full.capture_ids, full.captures):
        print(f"  {len(ds):5d}  {cid}")

    val_ids: list[str] = []
    if args.val_mode == "capture":
        if args.val_captures:
            val_ids = list(args.val_captures)
        else:
            rng = np.random.default_rng(args.seed)
            val_ids = list(rng.choice(full.capture_ids, size=min(args.n_val_captures,
                                                                  len(full.capture_ids)),
                                      replace=False))
        print(f"val mode: capture  held-out: {val_ids}")
        train_ds, val_ds = full.split_by_capture(val_ids)
    else:
        print(f"val mode: frame  val_frac={args.val_frac}")
        train_ds, val_ds = full.split_by_frame(args.val_frac, seed=args.seed)
    print(f"train frames: {len(train_ds)}  val frames: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("empty train or val split")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == "cuda"))

    model = TinyUNet(in_ch=args.n_frames, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Identity baseline: compare the center input frame to the target.
    center = args.n_frames // 2
    base_full, base_mask = [], []
    for batch in val_loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        x_center = x[:, center : center + 1]
        for i in range(x.size(0)):
            base_full.append(psnr(x_center[i], y[i]))
            base_mask.append(psnr(x_center[i], y[i], m[i]))
    base_full = float(np.mean(base_full))
    base_mask = float(np.mean(base_mask))
    print(f"baseline val PSNR (identity): full={base_full:.2f}  masked={base_mask:.2f}")

    aug_range = ((args.scale_aug_min, args.scale_aug_max)
                 if (args.scale_aug_min, args.scale_aug_max) != (1.0, 1.0) else None)
    if aug_range:
        print(f"scale aug: {aug_range[0]:.2f}-{aug_range[1]:.2f}x")

    history = {"train_loss": [], "val_loss": [],
               "val_psnr_full": [], "val_psnr_mask": []}
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            if aug_range is not None:
                x, y, m = random_scale_batch(x, y, m, aug_range)
            pred = model(x)
            loss = charbonnier(pred, y, m) + args.fft_weight * fourier_l1(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        vl, vf, vm = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                pred = model(x)
                loss = charbonnier(pred, y, m) + args.fft_weight * fourier_l1(pred, y)
                vl.append(float(loss.item()))
                for i in range(x.size(0)):
                    vf.append(psnr(pred[i], y[i]))
                    vm.append(psnr(pred[i], y[i], m[i]))
        tr = float(np.mean(train_losses))
        vlm = float(np.mean(vl))
        vfm = float(np.mean(vf))
        vmm = float(np.mean(vm))
        history["train_loss"].append(tr)
        history["val_loss"].append(vlm)
        history["val_psnr_full"].append(vfm)
        history["val_psnr_mask"].append(vmm)

        improved = vmm > best
        if improved:
            best = vmm
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args), "epoch": epoch, "val_psnr": vmm,
                        "val_captures": val_ids},
                       args.out / "best.pt")
        print(f"  epoch {epoch:3d}  train={tr:.4f}  val={vlm:.4f}  "
              f"PSNR full={vfm:.2f}  masked={vmm:.2f}  "
              f"Δmasked={vmm - base_mask:+.2f}" + ("  *" if improved else ""))

    print(f"done in {time.time() - t0:.1f}s  best masked PSNR: {best:.2f}  "
          f"(baseline {base_mask:.2f}, gain {best - base_mask:+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("masked loss"); axes[0].legend()
    axes[1].plot(history["val_psnr_mask"], label="model (masked)")
    axes[1].plot(history["val_psnr_full"], label="model (full)", alpha=0.5)
    axes[1].axhline(base_mask, color="C0", ls="--", alpha=0.6, label="identity (masked)")
    axes[1].axhline(base_full, color="C1", ls="--", alpha=0.6, label="identity (full)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val PSNR (dB)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=120)
    plt.close(fig)

    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    prev_dir = args.out / "previews"
    prev_dir.mkdir(exist_ok=True)
    n_show = min(12, len(val_ds))
    rng = np.random.default_rng(args.seed)
    pick_indices = rng.choice(len(val_ds), size=n_show, replace=False)
    with torch.no_grad():
        for k, vi in enumerate(pick_indices):
            sample = val_ds[int(vi)]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)
            pred = model(x)
            # Show the *center* input frame in the triptych so single- and
            # multi-frame runs are visually comparable.
            x_show = x[0, center : center + 1]
            row = np.concatenate([
                to_uint8_img(x_show),
                to_uint8_img(pred[0]),
                to_uint8_img(y[0]),
            ], axis=1)
            Image.fromarray(row).save(
                prev_dir / f"val_{k:02d}_{sample['capture_id']}_f{sample['frame_index']:06d}_q{sample['quality']:.2f}.png"
            )
    print(f"wrote {n_show} val triptychs to {prev_dir}")

    (args.out / "summary.json").write_text(json.dumps({
        "val_captures": val_ids,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "baseline_masked_psnr": base_mask,
        "baseline_full_psnr": base_full,
        "best_masked_psnr": best,
        "gain_db": best - base_mask,
    }, indent=2))


if __name__ == "__main__":
    main()
