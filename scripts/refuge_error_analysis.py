import argparse
import csv
from pathlib import Path


def load_predictions(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        pred_col = [c for c in reader.fieldnames if c.startswith("predicted_class")][0]
        for row in reader:
            rows.append({
                "filename": row["filename"],
                "ground_truth": int(row["ground_truth"]),
                "probability": float(row["predicted_probability"]),
                "predicted_class": int(row[pred_col]),
            })
    return rows


def main(args):
    rows = load_predictions(Path(args.predictions_csv))
    print(f"[INFO] Loaded {len(rows)} predictions")

    false_negatives = [r for r in rows if r["ground_truth"] == 1 and r["predicted_class"] == 0]
    false_positives = [r for r in rows if r["ground_truth"] == 0 and r["predicted_class"] == 1]
    true_positives = [r for r in rows if r["ground_truth"] == 1 and r["predicted_class"] == 1]
    true_negatives = [r for r in rows if r["ground_truth"] == 0 and r["predicted_class"] == 0]

    print(f"\n[SUMMARY]")
    print(f"  True Positives:  {len(true_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    print(f"  True Negatives:  {len(true_negatives)}")
    print(f"  False Positives: {len(false_positives)}")

    fn_sorted = sorted(false_negatives, key=lambda r: -r["probability"])
    fp_sorted = sorted(false_positives, key=lambda r: r["probability"])

    print(f"\n[FALSE NEGATIVES]")
    print(f"{'filename':<20} {'probability':>12}  {'distance from thresh':>20}")
    for r in fn_sorted:
        dist = r["probability"] - args.threshold
        print(f"{r['filename']:<20} {r['probability']:>12.4f}  {dist:>20.4f}")

    print(f"\n[FALSE POSITIVES] (first {args.max_display})")
    print(f"{'filename':<20} {'probability':>12}  {'distance from thresh':>20}")
    for r in fp_sorted[:args.max_display]:
        dist = r["probability"] - args.threshold
        print(f"{r['filename']:<20} {r['probability']:>12.4f}  {dist:>20.4f}")
    if len(fp_sorted) > args.max_display:
        print(f"... ({len(fp_sorted) - args.max_display} more, see CSV)")

    band = 0.10
    near_pos = [r for r in true_positives if abs(r["probability"] - args.threshold) <= band]
    near_neg = [r for r in true_negatives if abs(r["probability"] - args.threshold) <= band]
    near_bad = [r for r in (false_negatives + false_positives)
                if abs(r["probability"] - args.threshold) <= band]

    print(f"\n[NEAR-THRESHOLD] (within +/-{band} of {args.threshold})")
    print(f"  Correct TP: {len(near_pos)}, Correct TN: {len(near_neg)}, Incorrect: {len(near_bad)}")
    total_near = len(near_pos) + len(near_neg) + len(near_bad)
    if total_near > 0:
        print(f"  -> {len(near_bad)}/{total_near} ({100*len(near_bad)/total_near:.1f}%) are errors")

    overall_err = (len(false_negatives) + len(false_positives)) / len(rows)
    print(f"\n[COMPARISON] Overall error rate: {overall_err*100:.1f}%")

    outdir = Path(args.predictions_csv).parent
    for name, data in [("false_negatives", fn_sorted), ("false_positives", fp_sorted)]:
        outpath = outdir / f"refuge_{name}.csv"
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "probability", "predicted_class"])
            w.writeheader()
            for r in data:
                w.writerow(r)
        print(f"[INFO] Wrote {len(data)} cases to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-display", type=int, default=15)
    args = parser.parse_args()
    main(args)
