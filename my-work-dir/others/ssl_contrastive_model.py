"""
SSL Model Evaluation  (standalone — no project imports required)
================================================================
Computes, for the trained SSLResNet saved at MODEL_PATH:

  · Accuracy, Sn, Sp, TSS, FAR   (same metrics as meteor_detection.py)
  · Confusion matrix              (saved to OUTPUT_DIR)
  · Parameter count               (total / trainable)
  · Embedding dimensionality      (backbone h: 512-D, projector z: 256-D)
  · Inference time                (GPU + CPU, single image)

The classifier on top of the backbone is loaded from CLASSIFIER_PATH
(a joblib-serialised sklearn model or a pickled MLPClassifier).

Dataset layout expected under DATA_ROOT:
    DATA_ROOT/
        <filename>_CROP_SUMIMG.png   (or whatever suffix TEST_IMG_SUFFIX is)

TEST_CSV_PATH must be a semicolon-separated CSV with at least:
    · a filename column  (see FILENAME_COL)
    · a class column     (see CLASS_COL)  — values "meteor" / "non_meteor"
"""

import copy
import time
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH       = "../../../data/upftfg26/apujols/models/ssl_final_model_1.0.pt"
CLASSIFIER_PATH  = "../../../data/upftfg26/apujols/models/mlp_model_1.0.pt"
DATA_ROOT        = "../../../data/upftfg26/apujols/processed/original"
TEST_CSV_PATH    = "../../../data/upftfg26/apujols/datasets/dataset_test.csv"
OUTPUT_DIR       = "logs/training/ssl/"

FILENAME_COL     = "filename"        # column name with image names
CLASS_COL        = "class"           # column name with "meteor" / "unknown"
TEST_IMG_SUFFIX  = "_CROP_SUMIMG.png"  # suffix appended to filename to get the image path
CSV_SEP          = ";"

# Label convention: meteor=1, unknown(non_meteor)=0  (matches meteor_detection.py)
LABEL_MAP        = {"unknown": 0, "meteor": 1}
CLASS_NAMES      = ["meteor", "unknown"]

IMG_SIZE         = 128               # must match training resolution
BATCH_SIZE       = 32

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of warm-up / timed passes for inference-time measurement
WARMUP_RUNS      = 10
TIMING_RUNS      = 100


# ── Model definition (inlined — no project imports needed) ────────────────────

class SSLBackboneResNet(nn.Module):
    """ResNet18 backbone adapted for 1-channel input, classification head removed."""
    def __init__(self, res_net_dim=512):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        self.out_dim  = res_net_dim

    def forward(self, x):
        h = self.backbone(x)
        return h.squeeze(-1).squeeze(-1)   # (B, 512)


class SSLProjectionHeadSimCLR(nn.Module):
    """Two-layer projection head with BN (SimCLR style)."""
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, h):
        return F.normalize(self.net(h), dim=1)


class SSLResNet(nn.Module):
    """
    Full SSL model: ResNet18 encoder + SimCLR projection head.
    Backbone output h (512-D) is what the downstream classifier uses.
    Projector output z (256-D) is used during contrastive training only.
    """
    def __init__(self, res_net_dim=512, projection_dim=256):
        super().__init__()
        self.encoder   = SSLBackboneResNet(res_net_dim=res_net_dim)
        self.projector = SSLProjectionHeadSimCLR(in_dim=res_net_dim, out_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def encode_and_project(self, x):
        with torch.no_grad():
            h = self.encoder(x)
            z = self.projector(h)
        return h, z


# ── Classifier (MLPClassifier inlined for the case it was saved as one) ───────

class MLPClassifier(nn.Module):
    """Matches the MLPClassifier in classifiers.py exactly."""
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            device = next(self.parameters()).device
            X      = X.to(device)
            p1     = torch.sigmoid(self.forward(X)).cpu().numpy().flatten()
            return np.vstack([1 - p1, p1]).T


# ── Dataset ───────────────────────────────────────────────────────────────────

class NightSkyDataset(torch.utils.data.Dataset):
    """
    Minimal dataset that reads images listed in a CSV.
    Returns (tensor, label_int) — no auxiliary views needed for evaluation.
    """
    def __init__(self, csv_path, data_root, transform, filename_col, class_col,
                 label_map, img_suffix, sep=";"):
        self.df        = pd.read_csv(csv_path, sep=sep)
        self.data_root = Path(data_root)
        self.transform = transform
        self.fname_col = filename_col  # int index or string column name
        self.cls_col   = class_col     # int index or string column name
        self.label_map = label_map
        self.suffix    = img_suffix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.data_root / f"{row[self.fname_col]}{self.suffix}"
        img      = Image.open(img_path).convert("L")   # grayscale
        tensor   = self.transform(img)
        label    = self.label_map.get(row[self.cls_col], -1)
        if label == -1:
            raise ValueError(
                f"Unexpected class value '{row[self.cls_col]}' at row {idx}. "
                f"Update LABEL_MAP to include it."
            )
        return tensor, label


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            times = []
            for _ in range(runs):
                if use_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event   = torch.cuda.Event(enable_timing=True)
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


def compute_metrics(labels: np.ndarray, preds: np.ndarray):
    """
    Returns (metrics_dict, confusion_matrix).
    Positive class = meteor (label 1) — matches meteor_detection.py convention.
    """
    cm     = confusion_matrix(labels, preds, labels=[1, 0])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    acc = (TP + TN) / (TP + FP + FN + TN + 1e-9)
    sn  = TP / (TP + FN + 1e-9)
    sp  = TN / (TN + FP + 1e-9)
    tss = sn - FP / (FP + TN + 1e-9)
    far = FP / (FP + TP + 1e-9)

    return {"Acc": acc, "Sn": sn, "Sp": sp, "TSS": tss, "FAR": far}, cm


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

    path = Path(output_dir) / f"confusion_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def predict(clf, X, threshold=0.5):
    """Mirrors classifiers.predict() — works for sklearn and MLPClassifier."""
    y_probs = clf.predict_proba(X)

    if hasattr(clf, "classes_"):
        idx_meteor = np.where(clf.classes_ == 1)[0][0]
        p_meteor   = y_probs[:, idx_meteor]
    else:
        p_meteor = y_probs[:, 1]

    y_pred = (p_meteor > threshold).astype(int)
    return y_pred, y_probs


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluation():

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Output : {OUTPUT_DIR}")

    # ── Transform (must match training exactly) ───────────────────────────────
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.449], [0.226]),
    ])

    # ── Load SSL model ────────────────────────────────────────────────────────
    ssl_model = SSLResNet(res_net_dim=512, projection_dim=256)
    ssl_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    ssl_model.to(DEVICE)
    ssl_model.eval()
    print(ssl_model)

    # ── Load classifier ───────────────────────────────────────────────────────
    # mlp_model_1.0.pt was saved with joblib.dump on an MLPClassifier instance
    # whose fully-qualified class name is models.classifiers.MLPClassifier.
    # Registering the inlined class under that module path lets pickle resolve
    # it without the original project being on sys.path.
    import sys, types
    _fake_models   = types.ModuleType("models")
    _fake_clf_mod  = types.ModuleType("models.classifiers")
    _fake_clf_mod.MLPClassifier = MLPClassifier
    sys.modules["models"]             = _fake_models
    sys.modules["models.classifiers"] = _fake_clf_mod

    clf = joblib.load(CLASSIFIER_PATH)
    # Move the MLP to the correct device after loading
    if isinstance(clf, nn.Module):
        clf.to(DEVICE)
        clf.eval()
    print(f"\nClassifier loaded: {type(clf).__name__}")

    # ── 1. Parameter count ────────────────────────────────────────────────────
    total_params, trainable_params = count_parameters(ssl_model)
    print("\n── Parameter count ──────────────────────────────────────────────────")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    # ── 2. Embedding dimensionality ───────────────────────────────────────────
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        h_dummy, z_dummy = ssl_model.encode_and_project(dummy)
    print("\n── Embedding dimensionality ─────────────────────────────────────────")
    print(f"  Backbone  (h)        : {h_dummy.shape[1]}-D  {list(h_dummy.shape)}")
    print(f"  Projector (z)        : {z_dummy.shape[1]}-D  {list(z_dummy.shape)}")

    # ── 3. Inference time (backbone only — what the classifier uses) ──────────
    # Timing is measured on the encoder alone since the projector is discarded
    # at inference time, exactly as the classifier operates on h, not z.
    mean_dev, std_dev, mean_cpu, std_cpu = measure_inference_time(
        ssl_model.encoder, dummy)
    print(f"\n── Inference time — encoder only (single image, {IMG_SIZE}×{IMG_SIZE}) ──")
    print(f"  Warm-up runs         : {WARMUP_RUNS}")
    print(f"  Timed runs           : {TIMING_RUNS}")
    print(f"  Device ({str(DEVICE):<6})  Mean : {mean_dev:.3f} ms  Std : {std_dev:.3f} ms")
    print(f"  CPU          Mean : {mean_cpu:.3f} ms  Std : {std_cpu:.3f} ms")

    # ── 4. Extract embeddings on test set ─────────────────────────────────────
    dataset = NightSkyDataset(
        csv_path=TEST_CSV_PATH,
        data_root=DATA_ROOT,
        transform=base_transform,
        filename_col=FILENAME_COL,
        class_col=CLASS_COL,
        label_map=LABEL_MAP,
        img_suffix=TEST_IMG_SUFFIX,
        sep=CSV_SEP,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_h      = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            h    = ssl_model.encode(imgs)
            all_h.append(h.cpu().numpy())
            all_labels.append(labels.numpy())

    X      = np.vstack(all_h)
    y_true = np.concatenate(all_labels)

    print(f"\n  Test set             : {len(y_true)} images")
    print(f"  Label distribution   : {Counter(y_true.tolist())}")

    # ── 5. Classification metrics ─────────────────────────────────────────────
    y_pred, _ = predict(clf, X, threshold=0.5)
    metrics, cm = compute_metrics(y_true, y_pred)

    print("\n── Classification results ───────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:>5s}: {v:.4f}")
    print(f"  Prediction dist.     : {Counter(y_pred.tolist())}")

    save_confusion_matrix(cm, "SSL ResNet", OUTPUT_DIR)


def main():
    evaluation()


if __name__ == "__main__":
    main()