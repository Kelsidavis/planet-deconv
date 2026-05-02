"""Discover (input video, AS!3 unsharpened stack) pairs across drives and extract.

Discovery rules:
    PIPP chain:  <dir>/pipp_*/AS_P*/<base>_pipp_lapl5_*.tif
                  → input <dir>/pipp_*/<base>_pipp.ser
    Raw chain:   <dir>/AS_P*/<base>_lapl5_*.tif
                  → input <dir>/<base>.{ser,avi}

When multiple AS_P* (e.g. P50, P33) exist for one input, prefer P50, else
the largest percentile available.

Stacks containing "SHARPENING" in the filename are excluded — we want the
un-sharpened AS!3 output to avoid baking RegiStax wavelets into the targets.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from pair_extract import extract_one


# Strip the AS!3 "_lapl5_apNN_DrizzleNN" suffix from a TIF name to recover
# the source basename.
_AS_SUFFIX = re.compile(r"_lapl[0-9]+_ap[0-9]+_Drizzle[0-9]+\.tif$", re.IGNORECASE)
_AS_DIR = re.compile(r"^AS_P([0-9]+)$", re.IGNORECASE)


def find_unsharp_stacks(root: Path) -> list[Path]:
    """All AS!3 unsharpened TIFs under root."""
    out: list[Path] = []
    for p in root.rglob("*.tif"):
        parent = p.parent
        if not _AS_DIR.match(parent.name):
            continue
        if "sharpening" in p.name.lower():
            continue
        if not _AS_SUFFIX.search(p.name):
            continue
        out.append(p)
    return out


def stack_to_basename(stack: Path) -> str:
    """Strip the AS!3 suffix to recover the source basename."""
    return _AS_SUFFIX.sub("", stack.name)


def percentile_of_stack(stack: Path) -> int:
    m = _AS_DIR.match(stack.parent.name)
    return int(m.group(1)) if m else 0


def find_input_for(stack: Path) -> Path | None:
    """Locate the source video matching a given AS!3 stack TIF."""
    base = stack_to_basename(stack)
    cap_dir = stack.parent.parent  # the AS_P* directory's parent

    # PIPP chain: source is right next to AS_P* in the pipp_* dir.
    if cap_dir.name.lower().startswith("pipp_") or "_pipp" in base.lower():
        cand = cap_dir / f"{base}.ser"
        if cand.exists():
            return cand
        cand = cap_dir / f"{base}.avi"
        if cand.exists():
            return cand
        # PIPP-named TIF but file may live one dir up
        cand = cap_dir.parent / f"{base.removesuffix('_pipp')}.ser"
        if cand.exists():
            return cand
        cand = cap_dir.parent / f"{base.removesuffix('_pipp')}.avi"
        if cand.exists():
            return cand
        return None

    # Raw chain: source is one level up from AS_P*.
    for ext in (".ser", ".avi"):
        cand = cap_dir / f"{base}{ext}"
        if cand.exists():
            return cand
    return None


def best_stack_per_input(stacks_by_input: dict[Path, list[Path]]) -> dict[Path, Path]:
    """Prefer P50, else the largest percentile available, else any."""
    chosen: dict[Path, Path] = {}
    for inp, stacks in stacks_by_input.items():
        p50 = [s for s in stacks if percentile_of_stack(s) == 50]
        if p50:
            chosen[inp] = p50[0]
            continue
        chosen[inp] = max(stacks, key=percentile_of_stack)
    return chosen


def discover(roots: list[Path]) -> list[tuple[Path, Path]]:
    by_input: dict[Path, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for stack in find_unsharp_stacks(root):
            inp = find_input_for(stack)
            if inp is None:
                continue
            by_input.setdefault(inp, []).append(stack)
    chosen = best_stack_per_input(by_input)
    return [(inp, chosen[inp]) for inp in chosen]


def safe_capture_id(input_path: Path, all_pairs: list[tuple[Path, Path]]) -> str:
    """Disambiguate captures using parent dir + basename."""
    parent = input_path.parent.name
    stem = input_path.stem
    raw = f"{parent}__{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out-base", type=Path, default=Path("./pairs"))
    ap.add_argument("--manifest", type=Path, default=Path("./pairs/manifest.json"))
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--limit-per", type=int, default=0,
                    help="cap frames per capture (0 = all)")
    ap.add_argument("--max-captures", type=int, default=0,
                    help="process at most this many captures (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    args.out_base.mkdir(parents=True, exist_ok=True)
    pairs = discover(args.roots)
    print(f"discovered {len(pairs)} (input, stack) pairs")
    if args.max_captures > 0:
        pairs = pairs[: args.max_captures]
        print(f"limiting to {len(pairs)} captures")

    manifest: list[dict] = []
    t0 = time.time()
    for i, (inp, stack) in enumerate(pairs):
        cap_id = safe_capture_id(inp, pairs)
        out_dir = args.out_base / cap_id
        print(f"[{i+1}/{len(pairs)}] {cap_id}")
        print(f"  input: {inp}")
        print(f"  stack: {stack}")
        if args.dry_run:
            manifest.append({"capture_id": cap_id, "video": str(inp),
                             "stack": str(stack), "out": str(out_dir),
                             "kept": -1})
            continue
        if args.skip_existing and (out_dir / "inputs.npy").exists():
            print(f"  skip (exists)")
            continue
        try:
            stats = extract_one(
                inp, stack, out_dir,
                crop_size=args.crop_size, max_shift=args.max_shift,
                limit=args.limit_per, n_previews=4,
            )
            stats["capture_id"] = cap_id
            manifest.append(stats)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            manifest.append({"capture_id": cap_id, "video": str(inp),
                             "stack": str(stack), "out": str(out_dir),
                             "error": f"{type(e).__name__}: {e}"})

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    elapsed = time.time() - t0
    total_kept = sum(m.get("kept", 0) or 0 for m in manifest if isinstance(m.get("kept"), int))
    print(f"\ndone in {elapsed:.1f}s  total kept frames: {total_kept}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
