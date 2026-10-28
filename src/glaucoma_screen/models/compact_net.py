"""
Compact glaucoma referral classifier — architecture module.

Design rationale (applying the malaria-post lesson, not copying it):
- Malaria case proved: a single compact custom ResNet beat NIH's six-model
  pretrained ensemble on accuracy, at a fraction of size/training time,
  because the underlying classification task didn't need heavyweight
  ImageNet-scale feature extraction to separate two visually distinct classes.
- Glaucoma referral (fundus) is a HARDER problem than malaria cell tiles —
  subtle structural cues (rim thinning, RNFL defects, disc hemorrhage),
  continuous severity, not binary-obvious. So we don't go as small as the
  malaria model's 64x64 input — that would likely lose the fine structural
  detail JustRAIGS' own justification labels are built around.
- Compromise: 128x128 input (vs. Remidio's/typical full-res 224-299px),
  a ResNet-style architecture with FEWER stages/filters than ResNet-50,
  trained from scratch (no ImageNet pretraining dependency) — small enough
  for fast on-device inference, still enough capacity for real structural
  discrimination.
- This is a STARTING POINT, not a final answer. Actual input resolution
  and depth should be tuned empirically against a held-out validation set
  once real data is loaded — this is an architecture prior, not a claim.

Two outputs, matching the three-stage pipeline discussed:
  1. Quality gate (binary: sufficient / insufficient) — trained separately,
     placeholder here, needs its own labeled data (JustRAIGS excludes
     ungradable images from training set, so this needs another source —
     flagged as an open item, not solved by this file).
  2. Referral classifier (binary: RG / NRG) — this file's primary model.

Framework: PyTorch (chosen for straightforward ONNX export → mobile
deployment via ONNX Runtime Mobile or further conversion to TFLite).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block (BN -> ReLU -> Conv, x2)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(x))
        # shortcut branches off the *activated* tensor, standard pre-act design
        shortcut = self.shortcut(out) if self.shortcut is not None else identity
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + shortcut


class CompactGlaucomaNet(nn.Module):
    """
    Compact ResNet-style classifier for referable glaucoma screening.

    Input: 128x128x3 RGB fundus image (cropped/centered on optic disc
           upstream — this model assumes disc localization has already
           happened, it is NOT an end-to-end raw-image model).
    Output: single logit (binary: referable glaucoma probability via sigmoid)

    Stage config (filters, blocks per stage): much shallower than
    ResNet-50's (64,128,256,512 / 3-4-6-3 blocks) — start smaller, only
    grow if validation performance demands it.
    """

    def __init__(self, num_classes: int = 1, in_channels: int = 3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stages: (out_channels, num_blocks, stride)
        self.stage1 = self._make_stage(32, 32, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(128, 256, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logit — apply sigmoid at inference / BCEWithLogitsLoss during training


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check: instantiate, run a dummy forward pass, report size.
    model = CompactGlaucomaNet()
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    n_params = count_parameters(model)
    approx_size_mb = n_params * 4 / (1024 ** 2)  # float32, pre-quantization

    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {n_params:,}")
    print(f"Approx model size (fp32, pre-quantization): {approx_size_mb:.2f} MB")
