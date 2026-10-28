import argparse
import csv

import numpy as np

from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score


def clopper_pearson_ci(successes, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    if successes == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2, successes, n - successes + 1)
    if successes == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes)
    return (lower, upper)


def bootstrap_auc_ci(probs, labels, n_resamples=10000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    probs = np.array(probs)
    labels = np.array(labels)
    n = len(probs)

    boot_aucs = []
    skipped = 0
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_labels = labels[idx]
        boot_probs = probs[idx]
        if len(set(boot_labels)) < 2:
            skipped += 1
            continue
        boot_aucs.append(roc_auc_score(boot_labels, boot_probs))

    boot_aucs = np.array(boot_aucs)
    lower = np.percentile(boot_aucs, 100 * alpha / 2)
    upper = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return lower, upper, skipped, len(boot_aucs)


def main(args):
    filenames, labels, probs, preds = [], [], [], []
    with open(args.predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        pred_col = [c for c in reader.fieldnames if c.startswith("predicted_class")][0]
        for row in reader:
            filenames.append(row["filename"])
            labels.append(int(row["ground_truth"]))
            probs.append(float(row["predicted_probability"]))
            preds.append(int(row[pred_col]))

    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"[INFO] Loaded {n} predictions ({n_pos} positive, {n_neg} negative)")

    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(labels, probs)

    sens_lo, sens_hi = clopper_pearson_ci(tp, tp + fn)
    spec_lo, spec_hi = clopper_pearson_ci(tn, tn + fp)

    print(f"\n[RESULT] Point estimates:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f} (TP={tp}, FN={fn})")
    print(f"  Specificity: {specificity:.4f} (TN={tn}, FP={fp})")

    print(f"\n[RESULT] Exact (Clopper-Pearson) 95% confidence intervals:")
    print(f"  Sensitivity: {sensitivity:.4f} [{sens_lo:.4f}, {sens_hi:.4f}]  (n={tp+fn})")
    print(f"  Specificity: {specificity:.4f} [{spec_lo:.4f}, {spec_hi:.4f}]  (n={tn+fp})")

    print(f"\n[INFO] Bootstrap AUC CI: {args.n_resamples} resamples, seed={args.seed}")
    auc_lo, auc_hi, skipped, valid = bootstrap_auc_ci(
        probs, labels, n_resamples=args.n_resamples, seed=args.seed
    )
    print(f"[RESULT] Bootstrap 95% CI for AUC: {auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    if skipped > 0:
        print(f"[INFO] {skipped}/{args.n_resamples} resamples skipped "
              f"(only one class present). {valid} valid resamples used.")

    print(f"\n[SUMMARY]")
    print(f"  AUC: {auc:.4f} (95% CI: {auc_lo:.4f}-{auc_hi:.4f})")
    print(f"  Sensitivity: {sensitivity*100:.1f}% "
          f"(95% CI: {sens_lo*100:.1f}%-{sens_hi*100:.1f}%)")
    print(f"  Specificity: {specificity*100:.1f}% "
          f"(95% CI: {spec_lo*100:.1f}%-{spec_hi*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
