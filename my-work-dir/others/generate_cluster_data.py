"""
generate_cluster_data.py

Run this ONCE locally to produce assets/data/embeddings.json for the website.

Hardcoded paths — edit the CONFIGURATION section below before running.

Dataset CSV columns expected:
  - filename: stem of the image file (no extension)
  - class:    the true label — either "meteor" or one of the 13 non-meteor
              subclass names (airplane, bird, cosmic_ray, etc.)

Output JSON structure:
  {
    "points": [
      { "x": ..., "y": ..., "label": "airplane", "cluster": 3,
        "filename": "M20251004_...", "is_meteor": false }
    ],
    "subclasses":    ["airplane", "artificial_lights", ...],
    "cluster_profiles": { "0": { "dominant_label": "...", ... } },
    "k": 20,
    "n": 2591
  }
"""

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these paths before running
# ══════════════════════════════════════════════════════════════════

ENCODER_PATH    = "../../../data/upftfg26/apujols/models/ssl_final_model_1.0.pt"
KMEANS_PATH     = "../../../data/upftfg26/apujols/models/kmeans_k20.joblib"
DATA_CSV        = "../../../data/upftfg26/apujols/datasets/dataset_test_labeled.csv"
IMAGES_DIR      = "../../../data/upftfg26/apujols/processed/original"
OUTPUT_PATH     = "logs/embeddings.json"

# Column names in the CSV
FILE_COL        = "filename"
LABEL_COL       = "class"       # contains meteor OR subclass name directly

# Image filename suffix
IMAGE_EXT       = "_CROP_SUMIMG.png"

# t-SNE settings
TSNE_PERPLEXITY = 30
IMAGE_SIZE      = 255
MAX_POINTS      = 3000          # cap for t-SNE speed; set to None for all

# The meteor class name in your CSV
METEOR_LABEL    = "meteor"

# ══════════════════════════════════════════════════════════════════


# ── Model architecture (must match training exactly) ──────────────────────────

def get_resnet_backbone(backbone_dim):
    model = tv_models.resnet18(weights=None)
    if backbone_dim == 2048:
        model = tv_models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(*list(model.children())[:-1])


class SSLBackboneResNet(nn.Module):
    def __init__(self, res_net_dim):
        super().__init__()
        self.backbone = get_resnet_backbone(backbone_dim=res_net_dim)

    def forward(self, x):
        return self.backbone(x).squeeze(-1).squeeze(-1)


class SSLProjectionHeadSimCLR(nn.Module):
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
    def __init__(self, res_net_dim=512, projection_dim=256):
        super().__init__()
        self.encoder   = SSLBackboneResNet(res_net_dim)
        self.projector = SSLProjectionHeadSimCLR(res_net_dim, res_net_dim * 4, projection_dim)

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


# ── Image embedding ───────────────────────────────────────────────────────────

def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])


def embed_images(encoder, image_paths, device, batch_size=64):
    transform  = get_transform()
    encoder.eval()
    all_embeds = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('L')
                imgs.append(transform(img))
            except Exception as e:
                print(f"  Warning: could not load {p}: {e}")
                imgs.append(torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE))

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            h = encoder.encode(batch)
        all_embeds.append(h.cpu().numpy())

        done = min(i + batch_size, len(image_paths))
        if done % 500 == 0 or done == len(image_paths):
            print(f"  Embedded {done} / {len(image_paths)}")

    return np.vstack(all_embeds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load encoder ───────────────────────────────────────────
    print("\nLoading encoder...")
    encoder = SSLResNet(512, 256)
    state   = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    # ── 2. Load KMeans ────────────────────────────────────────────
    print("Loading KMeans...")
    kmeans = joblib.load(KMEANS_PATH)
    k      = kmeans.n_clusters
    print(f"  k = {k}")

    # ── 3. Load CSV ───────────────────────────────────────────────
    print("Loading CSV...")
    df = pd.read_csv(DATA_CSV, sep=';')
    print(f"  Total rows: {len(df)}")

    # The dataset already contains both meteor and non-meteor rows.
    # We visualise only non-meteor rows (the ones the clustering applies to).
    df_non = df[df[LABEL_COL] != METEOR_LABEL].copy().reset_index(drop=True)
    print(f"  Non-meteor rows: {len(df_non)}")

    if MAX_POINTS and len(df_non) > MAX_POINTS:
        df_non = df_non.sample(n=MAX_POINTS, random_state=42).reset_index(drop=True)
        print(f"  Capped to {MAX_POINTS} points")

    # ── 4. Build image paths ──────────────────────────────────────
    imgs_dir    = Path(IMAGES_DIR)
    image_paths = [imgs_dir / f"{row[FILE_COL]}{IMAGE_EXT}"
                   for _, row in df_non.iterrows()]

    missing = sum(1 for p in image_paths if not p.exists())
    if missing:
        print(f"  Warning: {missing} image files not found")

    # ── 5. Embed ──────────────────────────────────────────────────
    print("\nEmbedding images...")
    embeddings = embed_images(encoder, image_paths, device)
    print(f"  Shape: {embeddings.shape}")

    # ── 6. Cluster assignment ─────────────────────────────────────
    print("Assigning clusters...")
    cluster_ids = kmeans.predict(embeddings)

    # ── 7. t-SNE ──────────────────────────────────────────────────
    print(f"\nRunning t-SNE (perplexity={TSNE_PERPLEXITY}, n={len(embeddings)})...")
    tsne   = TSNE(n_components=3, perplexity=TSNE_PERPLEXITY,
                  random_state=42, max_iter=1000, verbose=1)
    coords = tsne.fit_transform(embeddings)
    print("  Done.")

    # ── 8. Load cluster profiles if available ─────────────────────
    profile_path = Path(KMEANS_PATH).parent / f"kmeans_k{k}_cluster_profiles.json"
    cluster_profiles = {}
    if profile_path.exists():
        with open(profile_path) as f:
            profile_data = json.load(f)
        cluster_profiles = profile_data.get('cluster_profiles', {})
        print(f"  Loaded cluster profiles from {profile_path}")
    else:
        print(f"  No cluster profiles found at {profile_path} — building from data")
        # Build minimal profiles from the data itself
        labels_arr = df_non[LABEL_COL].values
        for c in range(k):
            mask    = cluster_ids == c
            c_labels = labels_arr[mask]
            if len(c_labels) == 0:
                cluster_profiles[str(c)] = {'n_samples': 0, 'dominant_label': 'unknown'}
                continue
            unique, counts = np.unique(c_labels, return_counts=True)
            dominant = unique[np.argmax(counts)]
            profile  = {'n_samples': int(mask.sum()), 'dominant_label': str(dominant)}
            for lbl, cnt in zip(unique, counts):
                profile[str(lbl)] = round(float(cnt / mask.sum() * 100), 2)
            cluster_profiles[str(c)] = profile

    # ── 9. Build output JSON ──────────────────────────────────────
    labels      = df_non[LABEL_COL].values
    filenames   = df_non[FILE_COL].values
    subclasses  = sorted(set(labels))

    points = []
    for i in range(len(coords)):
        points.append({
            'x':        round(float(coords[i, 0]), 4),
            'y':        round(float(coords[i, 1]), 4),
            'z':        round(float(coords[i, 2]), 4),
            'label':    str(labels[i]),        # actual subclass name
            'cluster':  int(cluster_ids[i]),
            'filename': str(filenames[i]),
            'is_meteor': False,
        })

    output = {
        'points':           points,
        'subclasses':       subclasses,
        'clusters':         sorted(set(int(c) for c in cluster_ids)),
        'cluster_profiles': cluster_profiles,
        'k':                k,
        'n':                len(points),
    }

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f)

    print(f"\nSaved {len(points)} points → {out_path}")
    print(f"  Subclasses: {subclasses}")
    print(f"  Clusters:   {sorted(set(int(c) for c in cluster_ids))}")
    print("\nCopy the file to:  tfg-site/assets/data/embeddings.json")


if __name__ == '__main__':
    main()
