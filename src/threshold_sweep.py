"""
Threshold sweep for the trained CompactGlaucomaNet checkpoint.
Does NOT retrain -- one inference pass on the identical val split.
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

from glaucoma_screen.models.compact_net import CompactGlaucomaNet


class JustRAIGSDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str,
                 filename_col: str = "Eye ID", label_col: str = "Final Label"):
        self.image_dir = Path(image_dir)
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                label_value = row[label_col].strip().upper()
                if label_value not in ("RG", "NRG"):
                    continue
                label = 1 if label_value == "RG" else 0
                filename = f"{row[filename_col]}.JPG"
                self.samples.append((filename, label))

    def validate_files_exist(self):
        self.samples = [(fn, lbl) for fn, lbl in self.samples
                         if (self.image_dir / fn).exists()]

    def stratified_split(self, val_fraction: float, seed: int = 42):
        rng = random.Random(seed)
        pos = [s for s in self.samples if s[1] == 1]
        neg = [s for s in self.samples if s[1] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_val_pos = max(1, int(len(pos) * val_fraction))
        n_val_neg = max(1, int(len(neg) * val_fraction))
        val_samples = pos[:n_val_pos] + neg[:n_val_neg]
        train_samples = pos[n_val_pos:] + neg[n_val_neg:]
        rng.shuffle(val_samples)
        return train_samples, val_samples


class SplitView(Dataset):
    def __init__(self, samples, image_dir, transform):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().squeeze(1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    return all_probs, all_labels


def metrics_at_threshold(probs, labels, threshold):
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sensitivity, specificity, tp, fn, tn, fp


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = JustRAIGSDataset(args.csv_path, args.image_dir)
    print(f"[INFO] Loaded {len(dataset.samples)} samples from CSV")
    dataset.validate_files_exist()
    print(f"[INFO] {len(dataset.samples)} valid samples after file check")

    _, val_samples = dataset.stratified_split(val_fraction=args.val_fraction, seed=42)
    n_val_pos = sum(l for _, l in val_samples)
    print(f"[INFO] Val: {len(val_samples)} samples ({n_val_pos} positive) -- "
          f"identical split to training run")

    val_ds = SplitView(val_samples, Path(args.image_dir), val_transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))

    model = CompactGlaucomaNet().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    print("[INFO] Running inference on validation set (one pass, no retraining)...")
    probs, labels = get_predictions(model, val_loader, device)

    auc = roc_auc_score(labels, probs)
    print(f"\n[RESULT] Overall val_auc: {auc:.4f} (threshold-independent)\n")

    print(f"{'Threshold':>10} | {'Sensitivity':>11} | {'Specificity':>11} | "
          f"{'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5}")
    print("-" * 70)

    thresholds = [round(t, 2) for t in
                  [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]]
    results = []
    for t in thresholds:
        sens, spec, tp, fn, tn, fp = metrics_at_threshold(probs, labels, t)
        results.append((t, sens, spec))
        print(f"{t:>10.2f} | {sens:>11.4f} | {spec:>11.4f} | "
              f"{tp:>5} {fn:>5} {tn:>5} {fp:>5}")

    print()
    for t, sens, spec in results:
        if sens >= 0.90:
            print(f"[INFO] First threshold reaching >=90% sensitivity: "
                  f"{t} (sensitivity={sens:.4f}, specificity={spec:.4f})")
            break
    else:
        print("[INFO] No threshold in the swept range reached 90% sensitivity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
