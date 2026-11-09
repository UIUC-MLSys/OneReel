"""
eval/run_vbench.py

Standalone VBench evaluation script — run this in the VBench conda environment,
*separate* from the video-generation environment.

Usage
-----
# Evaluate all rows in an existing results.csv (writes vbench_* columns back in-place):
    conda run -n vbench python eval/run_vbench.py \
        --results-csv outputs/activation_ablation/results.csv \
        --dimensions imaging_quality aesthetic_quality motion_smoothness

# Evaluate one or more explicit video directories:
    conda run -n vbench python eval/run_vbench.py \
        --video-dirs outputs/activation_ablation/sweep1_fixed_window/act00_deact05 \
                     outputs/activation_ablation/sweep1_fixed_window/act05_deact10 \
        --output-dir outputs/activation_ablation/vbench_standalone

# Skip directories that already have vbench scores in the CSV:
    conda run -n vbench python eval/run_vbench.py \
        --results-csv outputs/activation_ablation/results.csv \
        --skip-done

Notes
-----
- This script calls VBench directly (no subprocess), so it must run in an
  environment that has `vbench` installed.
- If --results-csv is provided, vbench_* columns are written back into that
  file.  A backup is saved as results.csv.bak before modification.
- VBench writes its raw JSON output under <video_dir>/vbench_results/.
- dist_init() and VBench device setup are done ONCE at startup to avoid
  "process group initialised twice" errors across multiple directories.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from repo root
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))
from utils import load_csv, save_csv


# ---------------------------------------------------------------------------
# One-time VBench initialisation (called from main, not per directory)
# ---------------------------------------------------------------------------

_VBENCH_DEVICE = None
_VBENCH_FULL_JSON = None


def _init_vbench():
    """Initialise torch.distributed and resolve VBench's full_info JSON path.
    Must be called exactly once before any call to run_vbench_on_dir().
    """
    global _VBENCH_DEVICE, _VBENCH_FULL_JSON

    import typing
    import torch
    import torch.serialization

    # Allow typing.OrderedDict in safe-globals so VBench's AMT-S model loads
    # under PyTorch ≥ 2.6 (which defaults weights_only=True).
    try:
        torch.serialization.add_safe_globals([typing.OrderedDict])
    except AttributeError:
        pass  # older PyTorch — no-op

    from vbench.distributed import dist_init
    dist_init()

    _VBENCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import vbench as _vb_pkg
        _vb_dir = Path(_vb_pkg.__file__).parent
        _VBENCH_FULL_JSON = str(_vb_dir / "VBench_full_info.json")
    except Exception:
        _VBENCH_FULL_JSON = "./VBench_full_info.json"

    print(f"[VBench] device      : {_VBENCH_DEVICE}")
    print(f"[VBench] full_json   : {_VBENCH_FULL_JSON}")


# ---------------------------------------------------------------------------
# VBench runner (direct, no subprocess)
# ---------------------------------------------------------------------------

def run_vbench_on_dir(
    video_dir: str,
    dimensions: List[str],
    mode: str = "custom_input",
) -> Dict[str, float]:
    """
    Run VBench on *video_dir* and return {dimension: score}.
    Requires `vbench` to be importable in the current Python environment.
    _init_vbench() must have been called before this function.
    """
    from vbench import VBench

    # VBench's DINO evaluator does `from utils import trunc_normal_`, which
    # resolves to our eval/utils.py when eval/ is on sys.path.  Temporarily
    # hide eval/ and clear the sys.modules cache for "utils" so DINO can find
    # its own utils.py (inserted into sys.path from the torch hub cache).
    _eval_dir = str(Path(__file__).resolve().parent)
    _norm = os.path.normpath
    _removed_paths = [p for p in sys.path if _norm(p) == _norm(_eval_dir)]
    sys.path[:] = [p for p in sys.path if _norm(p) != _norm(_eval_dir)]
    _saved_utils = sys.modules.pop("utils", None)
    try:
        output_path = str(Path(video_dir) / "vbench_results")
        os.makedirs(output_path, exist_ok=True)

        my_vbench = VBench(_VBENCH_DEVICE, _VBENCH_FULL_JSON, output_path)
        name = f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

        my_vbench.evaluate(
            videos_path=video_dir,
            name=name,
            prompt_list=[],
            dimension_list=dimensions,
            local=False,
            read_frame=False,
            mode=mode,
        )

        # Parse JSON written by VBench
        scores: Dict[str, float] = {}
        for jf in sorted(Path(output_path).glob("results_*.json")):
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                for dim, vals in data.items():
                    if isinstance(vals, list) and len(vals) >= 1:
                        scores[dim] = float(vals[0])
                    elif isinstance(vals, (int, float)):
                        scores[dim] = float(vals)
            except Exception as e:
                print(f"  [VBench] Failed to parse {jf}: {e}", file=sys.stderr)

        return scores
    finally:
        # Restore eval/ on sys.path and our utils module
        if _saved_utils is not None:
            sys.modules["utils"] = _saved_utils
        for p in _removed_paths:
            if p not in sys.path:
                sys.path.insert(0, p)


def already_scored(row: Dict, dimensions: List[str]) -> bool:
    """True if every requested dimension already has a numeric score in *row*."""
    for dim in dimensions:
        val = row.get(f"vbench_{dim}", "")
        if not val or val in ("N/A", ""):
            return False
        try:
            float(val)
        except ValueError:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone VBench evaluator for activation ablation results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--results-csv", "--results_csv",
        type=str,
        default=None,
        help="Path to results.csv generated by test_activation.py.\n"
             "vbench_* columns are written back in-place.",
    )
    p.add_argument(
        "--video-dirs", "--video_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="One or more video directories to evaluate directly.\n"
             "Results are printed and saved to --output-dir.",
    )
    p.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        default="outputs/activation_ablation/vbench_standalone",
        help="Output directory for standalone results CSV (used with --video-dirs).",
    )
    p.add_argument(
        "--dimensions",
        nargs="+",
        default=["imaging_quality", "aesthetic_quality", "motion_smoothness", "subject_consistency", "background_consistency"],
        metavar="DIM",
        help="VBench dimensions to evaluate.",
    )
    p.add_argument(
        "--mode",
        default="custom_input",
        choices=["custom_input", "vbench_standard", "vbench_category"],
        help="VBench evaluation mode.",
    )
    p.add_argument(
        "--skip-done", "--skip_done",
        action="store_true",
        help="Skip CSV rows that already have all requested VBench scores.",
    )
    p.add_argument(
        "--eval-small-root", "--eval_small_root",
        type=str,
        default=None,
        metavar="DIR",
        help="Root of eval_small output (e.g. outputs/eval_small).\n"
             "Collects combined.mp4 from <root>/<setting>/prompt_*/combined.mp4\n"
             "and evaluates each setting. Use with --settings.",
    )
    p.add_argument(
        "--settings",
        nargs="+",
        default=["sp", "onereel"],
        metavar="SETTING",
        help="Settings to evaluate under --eval-small-root (default: sp onereel).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.results_csv is None and args.video_dirs is None and args.eval_small_root is None:
        print("ERROR: provide --results-csv, --video-dirs, or --eval-small-root.", file=sys.stderr)
        sys.exit(1)

    dimensions = args.dimensions
    print(f"[VBench] Dimensions: {dimensions}")
    print(f"[VBench] Mode      : {args.mode}")

    # Initialise torch.distributed and VBench metadata ONCE for the whole run.
    _init_vbench()

    # ── Mode A: evaluate video dirs listed in results.csv ────────────────────
    if args.results_csv is not None:
        csv_path = Path(args.results_csv)
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found.", file=sys.stderr)
            sys.exit(1)

        # Backup before modification
        bak = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, bak)
        print(f"[VBench] Backup written to {bak}")

        rows = load_csv(str(csv_path))
        total = len(rows)

        for i, row in enumerate(rows):
            vdir = row.get("video_dir", "")
            if not vdir or not Path(vdir).is_dir():
                print(f"  [{i+1}/{total}] SKIP — video_dir not found: {vdir!r}")
                continue

            if args.skip_done and already_scored(row, dimensions):
                print(f"  [{i+1}/{total}] SKIP (already scored) — {vdir}")
                continue

            act   = row.get("activation_step", "?")
            deact = row.get("deactivation_step", "?")
            print(f"\n  [{i+1}/{total}] act={act}  deact={deact}  → {Path(vdir).name}")

            try:
                scores = run_vbench_on_dir(vdir, dimensions, args.mode)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                scores = {}

            for dim, val in scores.items():
                row[f"vbench_{dim}"] = val
            print(f"    scores: {scores}")

            # Write back after every row so progress is not lost on crash
            save_csv(rows, str(csv_path))
            print(f"    [CSV] Updated {csv_path}")

        print(f"\n[VBench] Done. Results in {csv_path}")

    # ── Mode C: eval_small structured output (combined.mp4 per prompt) ───────
    elif args.eval_small_root is not None:
        eval_root = Path(args.eval_small_root)
        summary_rows: List[Dict] = []

        for setting in args.settings:
            setting_dir = eval_root / setting
            if not setting_dir.is_dir():
                print(f"  SKIP — setting dir not found: {setting_dir}")
                continue

            video_files = sorted(setting_dir.glob("prompt_*/combined.mp4"))
            if not video_files:
                print(f"  SKIP — no combined.mp4 found under {setting_dir}")
                continue

            print(f"\n[{setting}] {len(video_files)} videos found")

            # Build a flat directory of symlinks so VBench sees all videos at once.
            flat_dir = setting_dir / "_vbench_input"
            flat_dir.mkdir(exist_ok=True)
            for vf in video_files:
                link = flat_dir / f"{vf.parent.name}.mp4"  # e.g. prompt_0.mp4
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(vf.resolve())

            try:
                scores = run_vbench_on_dir(str(flat_dir), dimensions, args.mode)
            except Exception as e:
                print(f"  ERROR evaluating {setting}: {e}", file=sys.stderr)
                scores = {}

            row: Dict = {"setting": setting, "n_videos": len(video_files)}
            for dim, val in scores.items():
                row[f"vbench_{dim}"] = val
            summary_rows.append(row)
            print(f"  [{setting}] scores: {scores}")

            # Save after each setting so progress is not lost on crash.
            out_csv = eval_root / "vbench_summary.csv"
            save_csv(summary_rows, str(out_csv))
            print(f"  [CSV] Updated {out_csv}")

        print(f"\n[VBench] Done. Summary saved to {eval_root / 'vbench_summary.csv'}")

    # ── Mode B: evaluate explicit video dirs, write standalone CSV ───────────
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "vbench_results.csv"

        standalone_rows: List[Dict] = []

        for j, vdir in enumerate(args.video_dirs):
            if not Path(vdir).is_dir():
                print(f"  [{j+1}] SKIP — not a directory: {vdir!r}")
                continue

            print(f"\n  [{j+1}/{len(args.video_dirs)}] {vdir}")

            try:
                scores = run_vbench_on_dir(vdir, dimensions, args.mode)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                scores = {}

            row = {"video_dir": vdir}
            for dim, val in scores.items():
                row[f"vbench_{dim}"] = val
            standalone_rows.append(row)
            print(f"    scores: {scores}")

        save_csv(standalone_rows, str(out_csv))
        print(f"\n[VBench] Done. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
