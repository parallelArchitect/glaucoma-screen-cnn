"""
Build a small, stratified subset of JustRAIGS for cheap pipeline testing.
"""

import csv
import shutil
import random
import argparse
from pathlib import Path


def build_subset(justraigs_dir: Path, n_positive: int, n_negative: int, seed: int = 42):
    images_dir = justraigs_dir / "images"
    labels_csv = justraigs_dir / "JustRAIGS_Train_labels.csv"
    output_images_dir = justraigs_dir / "subset_images"
    output_labels_csv = justraigs_dir / "subset_labels.csv"

    output_images_dir.mkdir(exist_ok=True)

    samples = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            label = row["Final Label"].strip().upper()
            if label in ("RG", "NRG"):
                samples.append((row["Eye ID"] + ".JPG", label))

    print(f"[INFO] Loaded {len(samples)} labeled samples from CSV")

    pos = [s for s in samples if s[1] == "RG"]
    neg = [s for s in samples if s[1] == "NRG"]
    print(f"[INFO] {len(pos)} positive (RG), {len(neg)} negative (NRG) available")

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    actual_n_pos = min(n_positive, len(pos))
    if actual_n_pos < n_positive:
        print(f"[WARN] Requested {n_positive} positives but only {len(pos)} exist.")

    subset = pos[:actual_n_pos] + neg[:n_negative]
    rng.shuffle(subset)

    copied = 0
    missing = 0
    with open(output_labels_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Eye ID", "Final Label"])
        for filename, label in subset:
            src = images_dir / filename
            if not src.exists():
                missing += 1
                continue
            writer.writerow([filename.replace(".JPG", ""), label])
            shutil.copy(src, output_images_dir / filename)
            copied += 1

    print(f"[INFO] Subset built: {copied} images copied "
          f"({actual_n_pos} positive, {n_negative} negative requested), "
          f"{missing} missing/skipped")
    print(f"[INFO] Images: {output_images_dir}")
    print(f"[INFO] Labels: {output_labels_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--justraigs-dir", type=str,
                         default=str(Path.home() / "glaucoma-screen" / "data" / "raw" / "justraigs"))
    parser.add_argument("--n-positive", type=int, default=300)
    parser.add_argument("--n-negative", type=int, default=700)
    args = parser.parse_args()

    build_subset(Path(args.justraigs_dir), args.n_positive, args.n_negative)
