"""
Training script for CompactGlaucomaNet on JustRAIGS.

CONFIRMED schema:
- Delimiter is ';' semicolon, not comma.
- Eye ID + ".JPG" = actual filename, extracted flat into image_dir.
- Final Label values are exactly "RG" / "NRG".
- ~3.2% positive rate confirmed (pos_weight ~30 observed in first real run).

Evaluation:
Loss alone doesn't tell you whether an imbalanced-classification model is
actually useful. This script now holds out a validation split and reports,
every epoch: AUC, sensitivity, and specificity at 0.5 threshold. The
checkpoint with the BEST validation AUC is saved separately from the
final-epoch checkpoint.

Timing instrumentation:
Before the first real MI300X run, added wall-clock timing per epoch
(train/val samples-per-second, epoch time, projected full-run time) so
training length is measured, not guessed.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError(
        "scikit-learn is required for evaluation metrics. "
        "Install with: pip install scikit-learn"
    )

from glaucoma_screen.models.compact_net import CompactGlaucomaNet, count_parameters


class JustRAIGSDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str,
                 filename_col: str = "Eye ID", label_col: str = "Final Label"):
        self.image_dir = Path(image_dir)
        self.filename_col = filename_col
        self.label_col = label_col

        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            missing_cols = {filename_col, label_col} - set(reader.fieldnames or [])
            if missing_cols:
                raise ValueError(
                    f"Expected columns {missing_cols} not found in CSV. "
                    f"Actual columns: {reader.fieldnames}."
                )
            for row in reader:
                label_value = row[label_col].strip().upper()
                if label_value not in ("RG", "NRG"):
                    continue
                label = 1 if label_value == "RG" else 0
                filename = f"{row[filename_col]}.JPG"
                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def validate_files_exist(self, verbose: bool = True) -> None:
        missing = [fn for fn, _ in self.samples if not (self.image_dir / fn).exists()]
        if missing:
            if verbose:
                print(f"[WARN] {len(missing)} of {len(self.samples)} referenced "
                      f"images not found in {self.image_dir}. First 5: {missing[:5]}")
            self.samples = [(fn, lbl) for fn, lbl in self.samples
                             if (self.image_dir / fn).exists()]
            if verbose:
                print(f"[INFO] Proceeding with {len(self.samples)} valid samples.")

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
        rng.shuffle(train_samples)

        return train_samples, val_samples


class SplitView(Dataset):
    def __init__(self, samples, image_dir: Path, transform):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def compute_pos_weight(samples) -> torch.Tensor:
    labels = [label for _, label in samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive samples found — check label parsing.")
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().squeeze(1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    if len(set(all_labels)) < 2:
        return {"auc": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan")}

    auc = roc_auc_score(all_labels, all_probs)

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    tp = sum(1 for p, y in zip(preds, all_labels) if p == 1 and y == 1)
    fn = sum(1 for p, y in zip(preds, all_labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, all_labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, all_labels) if p == 1 and y == 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {"auc": auc, "sensitivity": sensitivity, "specificity": specificity}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = JustRAIGSDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
    )
    print(f"[INFO] Loaded {len(full_dataset)} samples from CSV")

    full_dataset.validate_files_exist()

    if args.limit_samples is not None:
        rng = random.Random(42)
        rng.shuffle(full_dataset.samples)
        full_dataset.samples = full_dataset.samples[:args.limit_samples]
        print(f"[INFO] Capped to {len(full_dataset)} samples for smoke test (shuffled)")

    train_samples, val_samples = full_dataset.stratified_split(
        val_fraction=args.val_fraction, seed=42
    )
    n_train_pos = sum(l for _, l in train_samples)
    n_val_pos = sum(l for _, l in val_samples)
    print(f"[INFO] Train: {len(train_samples)} samples ({n_train_pos} positive)")
    print(f"[INFO] Val:   {len(val_samples)} samples ({n_val_pos} positive)")

    image_dir = Path(args.image_dir)
    train_ds = SplitView(train_samples, image_dir, train_transform)
    val_ds = SplitView(val_samples, image_dir, val_transform)

    pos_weight = compute_pos_weight(train_samples)
    print(f"[INFO] Computed pos_weight for class imbalance: {pos_weight.item():.2f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = CompactGlaucomaNet().to(device)
    print(f"[INFO] Model parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    best_path = Path(args.output_path)
    final_path = best_path.with_name(best_path.stem + "_final" + best_path.suffix)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()

        model.train()
        running_loss = 0.0
        train_start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        train_time = time.perf_counter() - train_start

        scheduler.step()
        epoch_loss = running_loss / len(train_ds)

        val_start = time.perf_counter()
        metrics = evaluate(model, val_loader, device)
        val_time = time.perf_counter() - val_start

        epoch_time = time.perf_counter() - epoch_start
        train_rate = len(train_ds) / train_time if train_time > 0 else float("nan")
        val_rate = len(val_ds) / val_time if val_time > 0 else float("nan")

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss: {epoch_loss:.4f} | "
              f"val_auc: {metrics['auc']:.4f} | "
              f"val_sensitivity: {metrics['sensitivity']:.4f} | "
              f"val_specificity: {metrics['specificity']:.4f}")
        print(f"[PERF] "
              f"train_samples: {len(train_ds)} | "
              f"train_time: {train_time:.2f}s | "
              f"train_rate: {train_rate:.1f} img/s | "
              f"val_samples: {len(val_ds)} | "
              f"val_time: {val_time:.2f}s | "
              f"val_rate: {val_rate:.1f} img/s | "
              f"epoch_time: {epoch_time:.2f}s")

        if args.epochs > 1:
            projected_total_s = epoch_time * args.epochs
            print(f"[PERF] Projected total for {args.epochs} epochs at this rate: "
                  f"{projected_total_s/3600:.2f}h")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best val_auc {best_auc:.4f} — saved to {best_path}")

    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved final-epoch weights to {final_path}")
    print(f"[INFO] Best val_auc during training: {best_auc:.4f} (saved at {best_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--filename-col", type=str, default="Eye ID")
    parser.add_argument("--label-col", type=str, default="Final Label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    _default_output = str(
        Path(__file__).resolve().parents[3] / "models" / "checkpoints" / "glaucoma_model.pt"
    )
    parser.add_argument("--output-path", type=str, default=_default_output)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square classifier input resolution (default: 128)",
    )
    args = parser.parse_args()

    train(args)
