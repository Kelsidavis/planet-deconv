"""Train TinyUNet on one PairDataset and report against the identity baseline.

Usage:
    python train.py --pairs ./pairs/saturn_04_06_53_P50 --out ./runs/exp_001 \
        [--epochs 50] [--batch 16] [--lr 2e-4] [--val-frac 0.1]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import PairDataset
from model import TinyUNet


def charbonnier(pred: torch.Tensor, tgt: torch.Tensor,
                mask: torch.Tensor | None = None, eps: float = 1e-3) -> torch.Tensor:
    err = torch.sqrt((pred - tgt) ** 2 + eps * eps)
    if mask is None:
        return err.mean()
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def fourier_l1(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    P = torch.fft.rfft2(pred)
    T = torch.fft.rfft2(tgt)
    return (P.abs() - T.abs()).abs().mean()


def psnr(pred: torch.Tensor, tgt: torch.Tensor,
         mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mse = ((pred - tgt) ** 2).mean().item()
    else:
        m = mask
        mse = (((pred - tgt) ** 2) * m).sum().item() / max(float(m.sum().item()), 1.0)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def split_indices(n: int, val_frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def to_uint8_img(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().clamp(0, 1).numpy()
    if a.ndim == 3:
        a = a[0]
    return (a * 255.0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--fft-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base_ds = PairDataset(args.pairs)
    n = len(base_ds)
    train_idx, val_idx = split_indices(n, args.val_frac, args.seed)
    train_ds = PairDataset(args.pairs, train_idx)
    val_ds = PairDataset(args.pairs, val_idx)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0)

    model = TinyUNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Identity baseline (no model) on val: report whole-image and masked
    base_psnr_full = []
    base_psnr_mask = []
    for batch in val_loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        m = batch["mask"].to(device)
        for i in range(x.size(0)):
            base_psnr_full.append(psnr(x[i], y[i]))
            base_psnr_mask.append(psnr(x[i], y[i], m[i]))
    base_full = float(np.mean(base_psnr_full))
    base_mask = float(np.mean(base_psnr_mask))
    print(f"baseline val PSNR (identity): full={base_full:.2f} dB  masked={base_mask:.2f} dB")

    history = {"train_loss": [], "val_loss": [],
               "val_psnr_full": [], "val_psnr_mask": []}
    best_psnr = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            m = batch["mask"].to(device)
            pred = model(x)
            loss = charbonnier(pred, y, m) + args.fft_weight * fourier_l1(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        psnrs_full = []
        psnrs_mask = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                m = batch["mask"].to(device)
                pred = model(x)
                loss = charbonnier(pred, y, m) + args.fft_weight * fourier_l1(pred, y)
                val_losses.append(float(loss.item()))
                for i in range(x.size(0)):
                    psnrs_full.append(psnr(pred[i], y[i]))
                    psnrs_mask.append(psnr(pred[i], y[i], m[i]))

        tr = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        vp_full = float(np.mean(psnrs_full))
        vp_mask = float(np.mean(psnrs_mask))
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_psnr_full"].append(vp_full)
        history["val_psnr_mask"].append(vp_mask)

        improved = vp_mask > best_psnr
        if improved:
            best_psnr = vp_mask
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args), "epoch": epoch, "val_psnr": vp_mask},
                       args.out / "best.pt")
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  train={tr:.4f}  val={vl:.4f}  "
                  f"PSNR full={vp_full:.2f}  masked={vp_mask:.2f}  "
                  f"Δmasked={vp_mask - base_mask:+.2f}"
                  + ("  *" if improved else ""))

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  best masked val PSNR: {best_psnr:.2f} dB  "
          f"(baseline {base_mask:.2f}, gain {best_psnr - base_mask:+.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("masked loss")
    axes[0].legend()
    axes[1].plot(history["val_psnr_mask"], label="model (masked)")
    axes[1].plot(history["val_psnr_full"], label="model (full)", alpha=0.5)
    axes[1].axhline(base_mask, color="C0", ls="--", alpha=0.6, label="identity (masked)")
    axes[1].axhline(base_full, color="C1", ls="--", alpha=0.6, label="identity (full)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val PSNR (dB)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=120)
    plt.close(fig)

    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_show = min(6, len(val_ds))
    prev_dir = args.out / "previews"
    prev_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for k in range(n_show):
            sample = val_ds[k]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)
            pred = model(x)
            row = np.concatenate([
                to_uint8_img(x[0]),
                to_uint8_img(pred[0]),
                to_uint8_img(y[0]),
            ], axis=1)
            Image.fromarray(row).save(
                prev_dir / f"val_{k:02d}_frame{sample['frame_index']:06d}_q{sample['quality']:.2f}.png"
            )
    print(f"wrote {n_show} val triptychs (input | pred | target) to {prev_dir}")


if __name__ == "__main__":
    main()
