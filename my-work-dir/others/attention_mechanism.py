"""
Meteor Detection: VGG16 vs VGG16+CBAM  (PyTorch)
Based on: Shirasuna & Gradvohl, Astronomy and Computing 45 (2023) 100753

Usage:
    python meteor_detection.py --data_dir /path/to/dataset --csv_path /path/to/metadata.csv

Dataset structure expected:
    data_dir/
        train/
            meteor/
            non_meteor/
        val/
            meteor/
            non_meteor/
        test/
            meteor/
            non_meteor/

CSV expected columns:
    - filepath (or filename): path/name of the image
    - class: "meteor" or "non_meteor"

Outputs (saved to --output_dir, default ./results):
    - vgg16_best.pth            best VGG16 weights
    - vgg16_cbam_best.pth       best VGG16+CBAM weights
    - metrics.csv               Acc / Sn / Sp / TSS / FAR for both models
    - confusion_vgg16.png
    - confusion_vgg16_cbam.png
    - history_vgg16.csv
    - history_vgg16_cbam.csv
"""

import os
import copy
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ── Configuration — edit these before running ───────────────────────────────
DATA_DIR        = r"../../../data/upftfg26/apujols/state_of_the_art/nightskyUCP/"
CSV_PATH        = None
OUTPUT_DIR      = "logs/training/attention_mechanism/"
SKIP_TRAINING   = True

# ── Hyperparameters (paper Table 2) ─────────────────────────────────────────
IMG_SIZE        = 128
CROP_BOTTOM     = 0
TRAIN_EPOCHS    = 10
FINETUNE_EPOCHS = 10
TRAIN_BATCH     = 8
FINETUNE_BATCH  = 32
TRAIN_LR        = 1e-5
FINETUNE_LR     = 1e-4
WEIGHT_DECAY    = 5e-4
DROPOUT_RATE    = 0.2
DENSE_UNITS     = 16
CLASS_NAMES     = ["meteor", "non_meteor"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of warm-up passes before timing (avoids cold-start GPU overhead)
WARMUP_RUNS     = 10
# Number of timed passes for a stable inference-time estimate
TIMING_RUNS     = 100

# ── Transforms ───────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: img.crop(
        (0, 0, img.width, max(img.height - CROP_BOTTOM, 1)))),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226]),
])


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_loader(split_dir: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = datasets.ImageFolder(root=split_dir, transform=_transform)
    print(f"  {Path(split_dir).name}: {len(dataset)} images | "
          f"class mapping: {dataset.class_to_idx}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )


# ── CBAM ─────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """'What' branch of CBAM (Woo et al. ECCV 2018)."""
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * scale


class SpatialAttention(nn.Module):
    """'Where' branch of CBAM (Woo et al. ECCV 2018)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# ── Model Builders ───────────────────────────────────────────────────────────

def _vgg16_grayscale_backbone():
    """
    Load ImageNet-pretrained VGG16 and adapt the first conv layer to accept
    1-channel (grayscale) input by averaging the 3 RGB weight channels,
    preserving pretrained feature representations — same strategy as the
    existing ResNet18 code:
        new_weight = pretrained_weight.mean(dim=1, keepdim=True)
    """
    base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    old_conv = base.features[0]
    new_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=(old_conv.bias is not None))
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data.clone()
    base.features[0] = new_conv
    return base.features


class VGG16Meteor(nn.Module):
    """
    Optimised VGG16 (paper §3.3.1):
      ImageNet backbone (grayscale-adapted) → GlobalAveragePool →
      Dropout → Dense(16) → BN → Dropout → Sigmoid output
    """

    def __init__(self):
        super().__init__()
        self.features = _vgg16_grayscale_backbone()
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Flatten(),
            nn.Linear(512, DENSE_UNITS),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(DENSE_UNITS),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.gap(self.features(x)))

    def get_embedding(self, x):
        """Return the DENSE_UNITS-D feature vector (pre final linear+sigmoid)."""
        x = self.gap(self.features(x))
        x = self.head[0](x)   # Dropout
        x = self.head[1](x)   # Flatten
        x = self.head[2](x)   # Linear(512, DENSE_UNITS)
        x = self.head[3](x)   # ReLU
        x = self.head[4](x)   # BatchNorm1d
        return x               # shape: (batch, DENSE_UNITS)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


class VGG16CBAMMeteor(nn.Module):
    """
    VGG16+CBAM (paper §3.3.1, Fig. 2):
      CBAM inserted after each of the 5 VGG conv blocks (before each BN layer).
    """

    BLOCK_CHANNELS = [64, 128, 256, 512, 512]

    def __init__(self):
        super().__init__()
        layers = list(_vgg16_grayscale_backbone().children())

        blocks, cur = [], []
        for layer in layers:
            cur.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                blocks.append(nn.Sequential(*cur))
                cur = []

        self.blocks = nn.ModuleList(blocks)
        self.cbams  = nn.ModuleList(
            [CBAM(c) for c in self.BLOCK_CHANNELS]
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Flatten(),
            nn.Linear(512, DENSE_UNITS),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(DENSE_UNITS),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for block, cbam in zip(self.blocks, self.cbams):
            x = cbam(block(x))
        return self.head(self.gap(x))

    def get_embedding(self, x):
        """Return the DENSE_UNITS-D feature vector (pre final linear+sigmoid)."""
        for block, cbam in zip(self.blocks, self.cbams):
            x = cbam(block(x))
        x = self.gap(x)
        x = self.head[0](x)   # Dropout
        x = self.head[1](x)   # Flatten
        x = self.head[2](x)   # Linear(512, DENSE_UNITS)
        x = self.head[3](x)   # ReLU
        x = self.head[4](x)   # BatchNorm1d
        return x               # shape: (batch, DENSE_UNITS)

    def freeze_backbone(self):
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = True


# ── Metrics (Equations 1-5 from the paper) ───────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Returns (metrics_dict, confusion_matrix).
    Positive class = meteor (label 0).
    """
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    acc = (TP + TN) / (TP + FP + FN + TN + 1e-9)
    sn  = TP / (TP + FN + 1e-9)
    sp  = TN / (TN + FP + 1e-9)
    tss = sn - FP / (FP + TN + 1e-9)
    far = FP / (FP + TP + 1e-9)

    return {"Acc": acc, "Sn": sn, "Sp": sp, "TSS": tss, "FAR": far}, cm


# ── Model analysis helpers ────────────────────────────────────────────────────

def count_parameters(model):
    """Total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, input_tensor, warmup=WARMUP_RUNS, runs=TIMING_RUNS):
    """
    Returns mean and std of per-image inference time in milliseconds for both
    the original device and CPU.

    GPU timing uses CUDA events for accuracy (host-side perf_counter does not
    account for async kernel execution).  CPU timing moves the model and tensor
    to CPU regardless of the original device, so both figures are always
    reported even when running on GPU.  Warm-up passes are discarded in both
    cases to avoid cold-start overhead.
    """
    model.eval()

    def _time_on(m, x):
        use_cuda = x.device.type == "cuda"
        with torch.no_grad():
            for _ in range(warmup):
                m(x)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(runs):
                if use_cuda:
                    start_event.record()
                    m(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))   # ms
                else:
                    t0 = time.perf_counter()
                    m(x)
                    times.append((time.perf_counter() - t0) * 1e3)      # ms
        return float(np.mean(times)), float(np.std(times))

    # ── original device (GPU or CPU) ──────────────────────────────────────────
    mean_dev, std_dev = _time_on(model, input_tensor)

    # ── CPU ───────────────────────────────────────────────────────────────────
    cpu_model  = copy.deepcopy(model).to("cpu")
    cpu_tensor = input_tensor.to("cpu")
    mean_cpu, std_cpu = _time_on(cpu_model, cpu_tensor)

    return mean_dev, std_dev, mean_cpu, std_cpu


def print_model_analysis(model, name):
    """
    Print parameter count, embedding dimensionality, and inference time
    for a single model — called after weights are loaded and backbone
    is unfrozen so the trainable count reflects the deployed state.
    """
    # ── Parameter count ───────────────────────────────────────────────────────
    model.unfreeze_backbone()
    total_params, trainable_params = count_parameters(model)
    print(f"\n── {name} · Parameter count ──────────────────────────────────────")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    # ── Embedding dimensionality ──────────────────────────────────────────────
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    model.eval()
    with torch.no_grad():
        embedding = model.get_embedding(dummy)
    embed_dim = embedding.shape[1]
    print(f"\n── {name} · Embedding dimensionality ────────────────────────────")
    print(f"  Feature vector       : {embed_dim}-D  {list(embedding.shape)}")

    # ── Inference time ────────────────────────────────────────────────────────
    mean_dev, std_dev, mean_cpu, std_cpu = measure_inference_time(model, dummy)
    print(f"\n── {name} · Inference time (single image, {IMG_SIZE}×{IMG_SIZE}) ─────")
    print(f"  Warm-up runs         : {WARMUP_RUNS}")
    print(f"  Timed runs           : {TIMING_RUNS}")
    print(f"  Device ({str(DEVICE):<6})  Mean : {mean_dev:.3f} ms  Std : {std_dev:.3f} ms")
    print(f"  CPU          Mean : {mean_cpu:.3f} ms  Std : {std_cpu:.3f} ms")


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs    = imgs.to(DEVICE)
            targets = labels.float().unsqueeze(1).to(DEVICE)

            out  = model(imgs)
            loss = criterion(out, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            all_preds.append((out.detach().cpu() >= 0.5).long().squeeze(1))
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, output_dir, name):
    """
    Two-phase optimised training (paper §3.3.2):
      Phase 1 — frozen backbone  : 10 epochs, lr=1e-5, batch=8
      Phase 2 — full fine-tuning : 10 epochs, lr=1e-4, batch=32
    Best checkpoint chosen by lowest validation loss.
    """
    best_path      = Path(output_dir) / f"{name.replace('+','_')}_best.pth"
    best_val_loss  = float("inf")
    history        = []

    def _phase(phase_name, epochs, lr, batch_size):
        nonlocal best_val_loss
        loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY
        )

        for epoch in range(1, epochs + 1):
            tr_loss, tr_p, tr_l = run_epoch(model, loader, criterion, optimizer)
            vl_loss, vl_p, vl_l = run_epoch(model, val_loader, criterion)

            tr_m, _ = compute_metrics(tr_l, tr_p)
            vl_m, _ = compute_metrics(vl_l, vl_p)

            print(
                f"[{name}] {phase_name} {epoch:3d}/{epochs} | "
                f"Train  loss={tr_loss:.4f}  Acc={tr_m['Acc']:.4f} | "
                f"Val  loss={vl_loss:.4f}  Acc={vl_m['Acc']:.4f}  FAR={vl_m['FAR']:.4f}"
            )

            history.append({
                "phase": phase_name, "epoch": epoch,
                "train_loss": tr_loss, "val_loss": vl_loss,
                **{f"train_{k}": v for k, v in tr_m.items()},
                **{f"val_{k}":   v for k, v in vl_m.items()},
            })

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best weights → {best_path}")

    print(f"\n{'='*65}")
    print(f"  {name}  |  Phase 1: frozen backbone  "
          f"({TRAIN_EPOCHS} epochs, lr={TRAIN_LR}, bs={TRAIN_BATCH})")
    print(f"{'='*65}")
    model.freeze_backbone()
    _phase("phase1", TRAIN_EPOCHS, TRAIN_LR, TRAIN_BATCH)

    print(f"\n{'='*65}")
    print(f"  {name}  |  Phase 2: fine-tuning all layers  "
          f"({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR}, bs={FINETUNE_BATCH})")
    print(f"{'='*65}")
    model.unfreeze_backbone()
    _phase("finetune", FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH)

    return pd.DataFrame(history), str(best_path)


# ── Evaluation ────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fname = f"confusion_{model_name.lower().replace('+', '_')}.png"
    path  = Path(output_dir) / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {path}")


def evaluate_model(model, weights_path, test_loader, output_dir, name):
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.unfreeze_backbone()
    model.eval()

    # ── Classification metrics ────────────────────────────────────────────────
    _, preds, labels = run_epoch(model, test_loader, nn.BCELoss())
    metrics, cm      = compute_metrics(labels, preds)

    print(f"\n── {name} · Test Results ──────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:>5s}: {v:.4f}")

    save_confusion_matrix(cm, name, output_dir)

    # ── Parameter count, embedding dim, inference time ────────────────────────
    print_model_analysis(model, name)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Output : {OUTPUT_DIR}")

    if CSV_PATH:
        meta = pd.read_csv(CSV_PATH)
        print(f"\nCSV loaded: {len(meta)} rows  |  "
              f"columns: {list(meta.columns)}")
        print(meta["class"].value_counts().to_string())

    print("\nLoading splits:")
    if not SKIP_TRAINING:
        train_loader = make_loader(os.path.join(DATA_DIR, "my_train_images"),
                                   TRAIN_BATCH, shuffle=True)
        val_loader   = make_loader(os.path.join(DATA_DIR, "my_val_images"),
                                   FINETUNE_BATCH, shuffle=False)
    test_loader  = make_loader(os.path.join(DATA_DIR, "my_test_images"),
                               FINETUNE_BATCH, shuffle=False)

    if not SKIP_TRAINING:
        print(f"\nSplit sizes  ->  train: {len(train_loader.dataset)}  |  "
              f"val: {len(val_loader.dataset)}  |  test: {len(test_loader.dataset)}")
    else:
        print(f"\nSplit sizes  ->  test: {len(test_loader.dataset)}")

    criterion = nn.BCELoss()

    # Fixed paths to pre-trained weights — used when SKIP_TRAINING=True
    PRETRAINED_WEIGHTS = {
        "VGG16":      "../../../data/upftfg26/apujols/models/VGG16_best.pth",
        "VGG16+CBAM": "../../../data/upftfg26/apujols/models/VGG16_CBAM_best.pth",
    }

    configs = [
        ("VGG16",      VGG16Meteor()),
        ("VGG16+CBAM", VGG16CBAMMeteor()),
    ]

    all_metrics = {}
    for name, model in configs:
        model     = model.to(DEVICE)
        safe_name = name.replace("+", "_")

        if not SKIP_TRAINING:
            w_path = str(Path(OUTPUT_DIR) / f"{safe_name}_best.pth")
            history_df, w_path = train_model(
                model, train_loader, val_loader, criterion, OUTPUT_DIR, name)
            history_df.to_csv(
                Path(OUTPUT_DIR) / f"history_{safe_name.lower()}.csv",
                index=False)
        else:
            w_path = PRETRAINED_WEIGHTS[name]
            if not os.path.exists(w_path):
                raise FileNotFoundError(
                    f"SKIP_TRAINING=True but no weights found at: {w_path}"
                )
            print(f"\nLoading existing weights for {name}: {w_path}")

        all_metrics[name] = evaluate_model(
            model, w_path, test_loader, OUTPUT_DIR, name)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = (
        pd.DataFrame(all_metrics)
        .T.reset_index()
        .rename(columns={"index": "Model"})
    )
    summary_path = Path(OUTPUT_DIR) / "metrics.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*65)
    print("  Final Test Metrics Comparison")
    print("="*65)
    print(summary.to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
