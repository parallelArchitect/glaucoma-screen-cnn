import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma_screen.models.compact_net import CompactGlaucomaNet


class REFUGELabeledDataset(Dataset):
    def __init__(self, images_dir: Path, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            name = img_path.stem.lower()
            if name.startswith("g"):
                label = 1
            elif name.startswith("n"):
                label = 0
            else:
                continue
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def run_inference(model, loader, device):
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

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = REFUGELabeledDataset(args.images_dir, transform)
    print(f"[INFO] Loaded {len(dataset)} REFUGE images with g/n prefix labels")
    n_pos = sum(l for _, l in dataset.samples)
    print(f"[INFO] {n_pos} glaucoma (g), {len(dataset) - n_pos} normal (n)")

    if len(dataset) == 0:
        raise RuntimeError("No labeled images found.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    model = CompactGlaucomaNet().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded FROZEN checkpoint: {args.checkpoint}")

    print("[INFO] Running inference...")
    probs, labels = run_inference(model, loader, device)

    if len(set(labels)) < 2:
        print("[WARN] Only one class present -- AUC undefined.")
    else:
        auc = roc_auc_score(labels, probs)
        print(f"\n[RESULT] External (REFUGE) AUC: {auc:.4f}")

    sens, spec, tp, fn, tn, fp = metrics_at_threshold(probs, labels, args.threshold)
    print(f"[RESULT] At FROZEN threshold {args.threshold}:")
    print(f"  Sensitivity: {sens:.4f}  (TP={tp}, FN={fn})")
    print(f"  Specificity: {spec:.4f}  (TN={tn}, FP={fp})")
    print(f"\n[COMPARISON] JustRAIGS internal: sensitivity 93.27%, specificity 78.56%")
    print(f"[COMPARISON] REFUGE external: sensitivity {sens*100:.2f}%, specificity {spec*100:.2f}%")

    # Save raw per-sample predictions -- enables future ROC reconstruction,
    # confidence intervals, error analysis, calibration analysis without
    # rerunning inference.
    import csv
    output_csv = Path(args.checkpoint).parent / "external_val_refuge_predictions.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ground_truth", "predicted_probability",
                          f"predicted_class_at_{args.threshold}"])
        for (img_path, label), prob in zip(dataset.samples, probs):
            pred_class = 1 if prob >= args.threshold else 0
            writer.writerow([img_path.name, label, f"{prob:.6f}", pred_class])
    print(f"[INFO] Saved per-sample predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    main(args)
