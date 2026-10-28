"""
Objective pixel statistics comparison: REFUGE false negatives vs. true
positives. Real, computable stats -- no clinical interpretation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def compute_image_stats(img_path: Path):
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    grayscale = img.mean(axis=2)
    return {
        "mean_brightness": grayscale.mean(),
        "brightness_std": grayscale.std(),
        "mean_r": r.mean(),
        "mean_g": g.mean(),
        "mean_b": b.mean(),
        "min_brightness": grayscale.min(),
        "max_brightness": grayscale.max(),
    }


def summarize_group(stats_list, group_name):
    print(f"\n[{group_name}] n={len(stats_list)}")
    keys = stats_list[0].keys()
    summary = {}
    for key in keys:
        values = [s[key] for s in stats_list]
        mean = np.mean(values)
        std = np.std(values)
        summary[key] = (mean, std)
        print(f"  {key:>18}: {mean:>8.2f} +/- {std:>6.2f}")
    return summary


def main(args):
    images_dir = Path(args.images_dir)

    fn_filenames = []
    with open(args.false_negatives_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn_filenames.append(row["filename"])

    tp_filenames = []
    with open(args.predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        pred_col = [c for c in reader.fieldnames if c.startswith("predicted_class")][0]
        for row in reader:
            if int(row["ground_truth"]) == 1 and int(row[pred_col]) == 1:
                tp_filenames.append(row["filename"])

    print(f"[INFO] Computing stats for {len(fn_filenames)} false negatives "
          f"and {len(tp_filenames)} true positives...")

    fn_stats = [compute_image_stats(images_dir / f) for f in fn_filenames]
    tp_stats = [compute_image_stats(images_dir / f) for f in tp_filenames]

    fn_summary = summarize_group(fn_stats, "FALSE NEGATIVES (missed glaucoma)")
    tp_summary = summarize_group(tp_stats, "TRUE POSITIVES (correctly caught)")

    print(f"\n[COMPARISON] FN mean vs TP mean:")
    for key in fn_summary:
        fn_mean, fn_std = fn_summary[key]
        tp_mean, tp_std = tp_summary[key]
        diff = fn_mean - tp_mean
        pooled_std = np.sqrt((fn_std**2 + tp_std**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else float("nan")
        print(f"  {key:>18}: FN={fn_mean:>8.2f}  TP={tp_mean:>8.2f}  "
              f"diff={diff:>+8.2f}  effect_size={effect_size:>+6.2f}")

    print(f"\n[INTERPRETATION GUIDE] |effect_size| > 0.8 = large difference; "
          f"0.2-0.5 = small-to-medium; near 0 = no real measurable difference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--false-negatives-csv", type=str, required=True)
    parser.add_argument("--predictions-csv", type=str, required=True)
    args = parser.parse_args()

    main(args)
