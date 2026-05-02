# planet-deconv

A small ML pipeline for per-frame deconvolution of amateur planetary imagery. It
trains a U-Net to map individual short-exposure frames (or short bursts) of a
planet onto the corresponding lucky-imaging stack, using existing
PIPP / AutoStakkert!3 outputs as supervision.

The goal is a model that can run on raw `.ser` / `.avi` captures and produce
stack-quality output without keeping every frame from a session — useful as
either a preview during capture or a real-time deconvolution stage in a
processing pipeline.

## Pipeline

```
.ser / .avi capture            AS!3 unsharpened stack (.tif)
        │                                │
        ▼                                ▼
   per-frame                       single target image
   PIPP-centered
        │
        ▼
[ pair_extract.py / scale_extract.py ]
        │
        ▼
  per-capture dir:
    inputs.npy   (N, 256, 256) uint8
    target.npy   (256, 256)    uint8
    meta.npz     quality, shifts, frame_index
        │
        ▼
[ train_multi.py ]  →  TinyUNet checkpoint
        │
        ▼
[ infer.py ]  →  triptychs / per-frame restorations
```

## Files

| File | Purpose |
|---|---|
| `ser.py` | Parser for SER planetary capture files (random-access frames) |
| `video.py` | Unified video reader over SER + AVI (uses `imageio` + `pyav` for AVI) |
| `settings.py` | Parser for SharpCap `.CameraSettings.txt` sidecars |
| `crop.py` | Planet centroid + fixed-size crop helpers |
| `pair_extract.py` | Build `(frame, target)` training pairs from one capture |
| `scale_extract.py` | Discover all valid (input, stack) pairs across drives and extract |
| `inspect_capture.py` | Per-capture diagnostic: header, metadata, quality curve, previews |
| `dataset.py` | `PairDataset`, `MultiPairDataset` — per-capture normalization + planet mask |
| `model.py` | `TinyUNet` — multi-frame input, residual on the center frame |
| `train_multi.py` | Training loop with masked Charbonnier + Fourier loss, scale aug |
| `train.py` | Single-capture training loop (the original sanity baseline) |
| `infer.py` | Run a trained model on a fresh capture, optional triptych |

## Usage

### Install

```bash
pip install -e .
```

Required: `torch`, `numpy`, `matplotlib`, `pillow`, `imageio` (with `pyav` plugin).

### Discover and extract pairs across drives

```bash
python scale_extract.py \
    --roots /media/<drive>/path1 /media/<drive>/path2 \
    --out-base /path/to/pairs \
    --manifest /path/to/pairs/manifest.json \
    --limit-per 2000
```

Discovery rules:

- **PIPP chain**: `<dir>/pipp_*/AS_P*/<base>_pipp_lapl5_*.tif` → input `<dir>/pipp_*/<base>_pipp.ser`
- **Raw chain**: `<dir>/AS_P*/<base>_lapl5_*.tif` → input `<dir>/<base>.{ser,avi}`

Stacks containing `SHARPENING` in the filename are excluded — only the
unsharpened AS!3 output is used as the target, to avoid baking RegiStax /
WaveSharp wavelet behavior into the model.

### Train

```bash
python train_multi.py \
    --root /path/to/pairs \
    --out ./runs/exp_001 \
    --val-mode frame --val-frac 0.1 \
    --n-frames 5 \
    --scale-aug-min 0.5 --scale-aug-max 2.0 \
    --epochs 60 --batch 32 --workers 4
```

`--val-mode frame` holds out random frames across all captures (easier signal,
measures fit). `--val-mode capture` holds out entire captures (true
generalization test; harder).

### Run on a fresh capture

A pretrained checkpoint is included at `runs/best/best.pt` (TinyUNet, burst-5,
trained 60 epochs on 16 captures with scale aug 0.5–2.0×). Loss / PSNR curves
and val triptychs are in `runs/best/curves.png` and `runs/best/previews/`.

```bash
python infer.py \
    --video /path/to/capture.ser \
    --ckpt ./runs/best/best.pt \
    --out ./out \
    --stack /path/to/stack.tif    # optional, for triptych comparison
```

Saves a `frame_NNNNNN.png` per processed frame: `[input | model output | stack]`
if the stack is provided, otherwise `[input | model output]`.

## Design notes

A handful of decisions that turned out to matter, captured here so they don't
get lost:

- **Mask the loss/metric to the planet disk.** A 256×256 crop is ~95% black sky
  on a typical Saturn frame; whole-frame loss is dominated by trivially-matched
  zeros. Default mask: `target > 0.03` of normalized peak.
- **Per-capture normalization uses `input.max()`, not `target.max()`.** The
  same statistic must be computable at inference, where there's no target.
- **Residual head, not full-image regression.** The model predicts `center_input
  + residual`, which is much easier to learn than the full image when input is
  already close to target.
- **No output clamp during training.** `clamp(0,1)` saturates and zeros
  gradients on a poor init. Clamp only at inference (`model.predict`).
- **Scale augmentation 0.5–2.0× during training** is what makes the model
  generalize across captures with different disk-to-crop ratios. Without it the
  model memorizes "small bright disk on big dark crop" from Saturn-dominant
  data and produces dark halos / square edges on Jupiter-shaped scenes at
  inference.
- **Use `imageio` + `pyav` for AVI**, not `cv2`. `cv2.VideoCapture` segfaults on
  some uncompressed planetary AVIs (raw bgr24 from PIPP).
- **RGB SER frames must be collapsed to mono** before phase correlation.
  WinJUPOS DeRot output is `color_id=100` (RGB), and a 3D frame that bypasses
  the 2D assumption produces a `(256,256,2)` shape mismatch in `rfft2`.

## Status

Single-capture sanity loop works. Multi-capture training works on frame-level
holdout. Cross-capture generalization (held-out captures, true unseen
distributions) is the open problem — likely needs more data diversity, capture
conditioning, or both.

The repo is research code, not a polished tool — assume rough edges.
