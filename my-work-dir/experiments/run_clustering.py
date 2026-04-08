import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from experiments.run_classifier import get_split
from losses.losses import scan_loss
from models.modules import SCANHead


def build_knn_graph(X, k=20):
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return indices[:, 1:]  # remove self-neighbor


def initial_kmeans(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    labels = kmeans.fit_predict(X)
    return labels


def train_scan_head(X, knn_indices, init_labels, num_clusters, device="cuda", epochs=20):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    init_labels_t = torch.tensor(init_labels, dtype=torch.long).to(device)

    model = SCANHead(input_dim=X.shape[1], num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        logits = model(X_t)

        # nearest neighbor logits
        nn_logits = logits[knn_indices]  # shape: (N, k, C)
        nn_logits = nn_logits.mean(dim=1)  # average over neighbors

        loss = scan_loss(logits, nn_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model


def predict_scan_clusters(model, X, device="cuda"):
    """Simplified self-labeling step from SCAN"""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


def run_scan(cfg, num_clusters=20):
    # Load features
    X_train, X_val, y_train, y_val = get_split(cfg)
    X = np.concatenate([X_train, X_val], axis=0)

    # Step 1: kNN graph
    knn_indices = build_knn_graph(X, k=20)

    # Step 2: initial K-Means
    init_labels = initial_kmeans(X, num_clusters=num_clusters)

    # Step 3: train SCAN head
    model = train_scan_head(X, knn_indices, init_labels, num_clusters)

    # Step 4: final cluster assignments
    final_clusters = predict_scan_clusters(model, X)

    return final_clusters, model
