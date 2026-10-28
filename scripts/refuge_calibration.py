import argparse
import csv

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    labels = np.array(labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        bin_weight = n_in_bin / len(probs)
        ece += bin_weight * abs(avg_confidence - avg_accuracy)
        bin_details.append((lo, hi, n_in_bin, avg_confidence, avg_accuracy))

    return ece, bin_details


def main(args):
    labels, probs = [], []
    with open(args.predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["ground_truth"]))
            probs.append(float(row["predicted_probability"]))

    print("[INFO] Loaded " + str(len(labels)) + " predictions")

    brier = brier_score_loss(labels, probs)
    print("[RESULT] Brier score: {:.4f}".format(brier))
    print("[INFO] Lower is better. 0.25 = naive always-0.5 baseline on balanced data.")

    ece, bin_details = compute_ece(probs, labels, n_bins=args.n_bins)
    print("[RESULT] Expected Calibration Error (ECE): {:.4f}".format(ece))
    print("[INFO] ECE near 0 = well-calibrated. Higher = systematically over/under-confident.")

    print("[CALIBRATION BINS] (range | n | predicted | actual | direction)")
    for lo, hi, n, conf, acc in bin_details:
        if conf > acc:
            direction = "OVERCONFIDENT"
        elif conf < acc:
            direction = "UNDERCONFIDENT"
        else:
            direction = "MATCHED"
        print("  [{:.1f}-{:.1f}]  n={:>3}  predicted={:.3f}  actual={:.3f}  {}".format(
            lo, hi, n, conf, acc, direction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()
    main(args)
