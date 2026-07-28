"""One-time data-processing step: relabel every candlestick image by the
VISUAL trend it shows, instead of the dataset's original label (sign of the
5-day forward return AFTER the shown candles -- which we separately verified
carries ~no signal recoverable from the image, see notebook Section 4).

Usage:
    python scripts/relabel_by_visual_trend.py            # dry run, prints summary only
    python scripts/relabel_by_visual_trend.py --apply     # moves files + writes manifest CSV

The manifest (relabel_manifest.csv) logs every file's original label, new
label, and the measured slope, for auditability -- this is a real change to
ground truth, not a cosmetic one, so it should be traceable.
"""

import argparse
import csv
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import detect_visual_trend  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually move files")
    parser.add_argument("--manifest", default="relabel_manifest.csv")
    args = parser.parse_args()

    manifest_rows = []

    for split in ("train", "test"):
        print(f"\n===== {split} =====")

        # Phase 1: snapshot every (path, original_label) and compute its
        # visual-trend label BEFORE moving anything. Moving files while still
        # scanning the source directories would let a file moved into e.g.
        # Down on this pass get rescanned as if it started there, double-
        # counting it and corrupting the stats.
        to_process = []
        for original_label in ("Up", "Down"):
            for p in sorted(glob.glob(f"data/{split}/{original_label}/*.png")):
                to_process.append((p, original_label))

        agree = 0
        new_counts = {"Up": 0, "Down": 0}
        moves = []  # (src, dest) pairs, executed after the full scan

        for p, original_label in to_process:
            new_label, slope = detect_visual_trend(p)
            new_counts[new_label] += 1
            if new_label == original_label:
                agree += 1

            filename = os.path.basename(p)
            manifest_rows.append(
                {
                    "split": split,
                    "filename": filename,
                    "original_label": original_label,
                    "visual_trend_label": new_label,
                    "slope": round(slope, 4),
                }
            )

            if new_label != original_label:
                # Filenames are only unique *within* their original label
                # folder in this dataset (it spans multiple tickers, e.g.
                # QQQ_5.png and a completely different SPY-derived image can
                # both be named QQQ_5.png... no -- concretely: the same
                # basename can independently exist in both Up/ and Down/).
                # Prefixing with the original label on move guarantees the
                # destination filename can never collide with a file already
                # there, which silently overwrote ~291 images last attempt.
                dest_dir = f"data/{split}/{new_label}"
                dest_path = os.path.join(dest_dir, f"{original_label}_{filename}")
                moves.append((p, dest_dir, dest_path))

        total = len(to_process)
        print(f"Total images: {total}")
        print(f"New visual-trend class split: {new_counts}")
        print(f"Agreement with original (future-return) label: {agree}/{total} = {agree/total:.1%}")
        print(f"Files to move: {len(moves)}")

        if args.apply:
            for src, dest_dir, dest_path in moves:
                os.makedirs(dest_dir, exist_ok=True)
                assert not os.path.exists(dest_path), f"collision: {dest_path}"
                shutil.move(src, dest_path)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "filename", "original_label", "visual_trend_label", "slope"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest written to {args.manifest} ({len(manifest_rows)} rows)")

    if args.apply:
        print("Files moved -- data/train and data/test now reflect visual-trend labels.")
    else:
        print("Dry run only -- rerun with --apply to actually move files.")


if __name__ == "__main__":
    main()
