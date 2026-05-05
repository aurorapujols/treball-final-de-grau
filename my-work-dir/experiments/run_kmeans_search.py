import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.preprocessing import normalize

import models.ssl_model as encoder
import data.datasets as datasets
import transformations.transform as transform
from data.dataloaders import get_ssl_loader


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def extract_embeddings(cfg, split="train+val"):
    """Extract and L2-normalize backbone embeddings.
    
    Returns
    -------
    X_norm      : (N, D) normalized embeddings
    y           : (N,) binary labels  0=non-meteor  1=meteor
    fine_labels : (N,) raw string labels from the dataloader (e.g. "cosmic_ray", "meteor", ...)
                  Returns None if labels are only binary (unknown/meteor).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERSION = cfg["experiment_version"]
    batch_size = cfg.get("batch_size", 128)

    ssl_model = encoder.get_model(cfg["ssl_model_path"])
    ssl_model.to(device)
    ssl_model.eval()

    train_set, val_set, _ = datasets.get_dataset_split(
        full_dataset_csv_path=cfg["paths"]["full_dataset"],
        output_path=cfg["paths"]["datasets_dir"]
    )

    if split == "train+val":
        import pandas as pd
        df = pd.concat([train_set, val_set], ignore_index=True)
    else:
        df = {"train": train_set, "val": val_set}[split]

    _, loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        dataframe=df,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION,
        shuffle=False
    )

    X, _, y_raw = encoder.get_encoding_and_projection(
        model=ssl_model, dataloader=loader, device=device
    )

    # Binary labels
    label_map = {"unknown": 0, "meteor": 1}
    y = np.array([label_map.get(lbl, 0) for lbl in y_raw], dtype=np.int64)

    # Fine-grained labels — keep as strings if they carry class info
    fine_labels = np.array(y_raw)
    unique = set(y_raw)
    # If only binary labels present, fine-grained heatmaps won't be meaningful
    if unique <= {"unknown", "meteor"}:
        fine_labels = None

    X_norm = normalize(X, norm="l2")

    return X_norm, y, fine_labels


def filter_non_meteors(X, y):
    """Return only the non-meteor embeddings."""
    mask = y == 0
    return X[mask]


# ----------------------------------------------------------------
# K search
# ----------------------------------------------------------------

def search_k(
    X,
    k_range=range(2, 31),
    n_init=20,
    random_state=42,
    use_gmm=False
):
    """
    For each K, fit K-Means (or GMM) and compute:
      - Inertia (K-Means only)
      - Silhouette Score
      - Davies-Bouldin Index

    Parameters
    ----------
    X         : (N, D) normalized embeddings
    k_range   : range of K values to evaluate
    n_init    : number of K-Means restarts (higher = more stable)
    use_gmm   : if True, also fits a GMM and records BIC/AIC

    Returns
    -------
    results : dict with lists of scores per K
    """
    k_values     = list(k_range)
    inertias     = []
    silhouettes  = []
    davies       = []
    bic_scores   = []
    aic_scores   = []

    for k in k_values:
        print(f"  K={k:3d} ...", end=" ", flush=True)

        # --- K-Means ---
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

        # Silhouette and Davies-Bouldin need at least 2 clusters
        sil = silhouette_score(X, labels, sample_size=5000, random_state=random_state)
        db  = davies_bouldin_score(X, labels)
        silhouettes.append(sil)
        davies.append(db)

        # --- GMM (optional) ---
        if use_gmm:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",   # faster than "full" for high-dim
                n_init=3,
                random_state=random_state
            )
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            print(f"sil={sil:.4f}  db={db:.4f}  bic={gmm.bic(X):.1f}")
        else:
            print(f"sil={sil:.4f}  db={db:.4f}  inertia={km.inertia_:.1f}")

    results = {
        "k":            k_values,
        "inertia":      inertias,
        "silhouette":   silhouettes,
        "davies_bouldin": davies,
    }
    if use_gmm:
        results["bic"] = bic_scores
        results["aic"] = aic_scores

    return results


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

def plot_k_search(results, title_suffix="", save_path=None, use_gmm=False):
    """
    Plots all K-selection metrics in a single figure.
    The best K according to each metric is marked with a vertical dashed line.
    """
    k        = results["k"]
    sil      = results["silhouette"]
    db       = results["davies_bouldin"]
    inertia  = results["inertia"]

    best_k_sil = k[np.argmax(sil)]
    best_k_db  = k[np.argmin(db)]

    n_plots = 4 if use_gmm else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle(f"K-Means Cluster Search  {title_suffix}", fontsize=13)

    # --- Elbow (Inertia) ---
    ax = axes[0]
    ax.plot(k, inertia, "o-", color="steelblue")
    ax.set_title("Elbow (Inertia)")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)

    # --- Silhouette ---
    ax = axes[1]
    ax.plot(k, sil, "o-", color="darkorange")
    ax.axvline(best_k_sil, linestyle="--", color="darkorange", alpha=0.6,
               label=f"best K={best_k_sil}")
    ax.set_title("Silhouette Score  (↑ better)")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Davies-Bouldin ---
    ax = axes[2]
    ax.plot(k, db, "o-", color="seagreen")
    ax.axvline(best_k_db, linestyle="--", color="seagreen", alpha=0.6,
               label=f"best K={best_k_db}")
    ax.set_title("Davies-Bouldin Index  (↓ better)")
    ax.set_xlabel("K")
    ax.set_ylabel("DB Index")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- BIC/AIC (GMM only) ---
    if use_gmm and "bic" in results:
        ax = axes[3]
        ax.plot(k, results["bic"], "o-", color="purple", label="BIC")
        ax.plot(k, results["aic"], "s--", color="crimson", label="AIC")
        best_k_bic = k[np.argmin(results["bic"])]
        ax.axvline(best_k_bic, linestyle=":", color="purple", alpha=0.6,
                   label=f"best BIC K={best_k_bic}")
        ax.set_title("GMM BIC / AIC  (↓ better)")
        ax.set_xlabel("K")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def print_summary(results):
    k   = results["k"]
    sil = results["silhouette"]
    db  = results["davies_bouldin"]

    best_k_sil = k[np.argmax(sil)]
    best_k_db  = k[np.argmin(db)]

    print("\n========== K Search Summary ==========")
    print(f"  Best K by Silhouette Score : K={best_k_sil}  (score={max(sil):.4f})")
    print(f"  Best K by Davies-Bouldin   : K={best_k_db}  (score={min(db):.4f})")
    if "bic" in results:
        best_k_bic = k[np.argmin(results["bic"])]
        print(f"  Best K by GMM BIC          : K={best_k_bic}")
    print("======================================\n")


# ----------------------------------------------------------------
# Silhouette diagram for a specific set of K values
# ----------------------------------------------------------------

def plot_silhouette_diagrams(X, k_values, n_init=20, random_state=42,
                              save_path=None, title_suffix=""):
    """
    For each K in k_values, plots a silhouette diagram showing:
      - Left panel: per-point silhouette scores grouped by cluster (the 'blade' plot)
      - Right panel: a 2D PCA projection of the clusters for visual inspection

    Parameters
    ----------
    X          : (N, D) normalized embeddings
    k_values   : list of K values to plot, e.g. [5, 9, 15, 20]
    save_path  : base path; each K saved as {save_path}_K{k}.png
    """
    from sklearn.decomposition import PCA

    # Reduce to 2D once for all plots (for the scatter panel)
    print("  Computing 2D PCA projection for silhouette diagrams...")
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    for k in k_values:
        print(f"  Silhouette diagram K={k} ...", end=" ", flush=True)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)

        # Per-point silhouette values
        sil_vals = silhouette_samples(X, labels)
        avg_sil  = sil_vals.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Silhouette Diagram  K={k}  {title_suffix}\n"
            f"Mean silhouette = {avg_sil:.4f}",
            fontsize=12
        )

        # --- Left: silhouette blades ---
        colors = cm.tab20(np.linspace(0, 1, k))
        y_lower = 10
        for i in range(k):
            cluster_sil = np.sort(sil_vals[labels == i])
            size = len(cluster_sil)
            y_upper = y_lower + size

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, cluster_sil,
                facecolor=colors[i], edgecolor=colors[i], alpha=0.8
            )
            # Cluster label in the middle of the blade
            ax1.text(-0.05, y_lower + size / 2, str(i), fontsize=7)
            y_lower = y_upper + 10  # gap between clusters

        ax1.axvline(avg_sil, color="red", linestyle="--", lw=1.5,
                    label=f"mean={avg_sil:.3f}")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_title("Per-point silhouette values")
        ax1.set_xlim([-0.1, 1.0])
        ax1.set_yticks([])
        ax1.legend(fontsize=9)

        # --- Right: 2D PCA scatter ---
        for i in range(k):
            mask = labels == i
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        color=colors[i], s=4, alpha=0.5, label=f"C{i}")
            # Mark centroid
            cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
            ax2.text(cx, cy, str(i), fontsize=8, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.2", fc="white", alpha=0.6))

        ax2.set_xlabel(f"PC1")
        ax2.set_ylabel(f"PC2")
        ax2.set_title(f"2D PCA projection  ({var_explained:.1f}% variance)")
        ax2.grid(alpha=0.2)

        plt.tight_layout()

        if save_path:
            path = f"{save_path}_K{k}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved → {path}")
        else:
            print()

        plt.close(fig)


# ----------------------------------------------------------------
# Purity heatmap
# ----------------------------------------------------------------

def compute_purity_matrix(cluster_labels, true_labels):
    """
    Builds a (n_clusters x n_classes) matrix where each cell [i, j]
    is the percentage of points in cluster i that belong to true class j.
    Rows sum to 100%.

    Parameters
    ----------
    cluster_labels : (N,) int array  — K-Means cluster assignments
    true_labels    : (N,) string/int array — ground truth fine-grained labels

    Returns
    -------
    purity_matrix : (n_clusters, n_classes) float array  (percentages)
    cluster_names : list of cluster row labels e.g. ["C0", "C1", ...]
    class_names   : list of unique true class names (columns)
    """
    unique_clusters = np.unique(cluster_labels)
    unique_classes  = np.unique(true_labels)

    purity = np.zeros((len(unique_clusters), len(unique_classes)), dtype=float)

    for i, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        total = mask.sum()
        for j, cls in enumerate(unique_classes):
            purity[i, j] = (true_labels[mask] == cls).sum() / total * 100

    cluster_names = [f"C{c}" for c in unique_clusters]
    class_names   = list(unique_classes)

    return purity, cluster_names, class_names


def plot_purity_heatmap(purity_matrix, cluster_names, class_names,
                        title="Cluster Purity Heatmap", save_path=None):
    """
    Replicates your existing plot_confusion_matrix_heatmap style
    but for cluster purity (percentage, not counts).
    Clusters are sorted by their dominant class for easier reading.
    """
    # Sort clusters by the column index of their dominant class
    dominant = np.argmax(purity_matrix, axis=1)
    sort_idx = np.argsort(dominant, kind="stable")
    purity_sorted   = purity_matrix[sort_idx]
    clusters_sorted = [cluster_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8),
                                    max(8,  len(cluster_names) * 0.45)))

    im = ax.imshow(purity_sorted, cmap="viridis", vmin=0, vmax=100, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(clusters_sorted)))
    ax.set_xticklabels(class_names, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(clusters_sorted, fontsize=11)

    ax.set_xlabel("True Class", fontsize=13)
    ax.set_ylabel("Cluster", fontsize=13)
    ax.set_title(title, fontsize=14)

    # Annotate cells — only show values > 5% to avoid clutter
    for i in range(purity_sorted.shape[0]):
        for j in range(purity_sorted.shape[1]):
            val = purity_sorted[i, j]
            if val > 5:
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if val > 50 else "black")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")

    plt.close(fig)
    return fig


def generate_purity_heatmaps(X, fine_labels, k_values, n_init=20,
                              random_state=42, save_path=None, title_suffix=""):
    """
    For each K in k_values, fits K-Means and plots the purity heatmap.

    Parameters
    ----------
    X           : (N, D) normalized embeddings
    fine_labels : (N,) array of fine-grained string labels (e.g. "meteor", "cosmic_ray", ...)
    k_values    : list of K values to evaluate
    save_path   : base path; files saved as {save_path}_K{k}.png
    """
    for k in k_values:
        print(f"  Purity heatmap K={k} ...", end=" ", flush=True)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        cluster_labels = km.fit_predict(X)

        purity, cluster_names, class_names = compute_purity_matrix(
            cluster_labels, fine_labels
        )

        sp = f"{save_path}_K{k}.png" if save_path else None
        plot_purity_heatmap(
            purity, cluster_names, class_names,
            title=f"Cluster Purity Heatmap  K={k}  {title_suffix}",
            save_path=sp
        )


# ----------------------------------------------------------------
# Entry point — called from main.py
# ----------------------------------------------------------------

def run_k_search(cfg):
    """
    Main function. Call with:
        python main.py --task k_search --config experiments/clustering.yaml
    """
    output_path = cfg["paths"]["output_dir"]
    use_gmm     = cfg.get("k_search", {}).get("use_gmm", False)
    k_min       = cfg.get("k_search", {}).get("k_min", 2)
    k_max       = cfg.get("k_search", {}).get("k_max", 30)
    n_init      = cfg.get("k_search", {}).get("n_init", 20)
    sil_ks      = cfg.get("k_search", {}).get("sil_diagram_ks", [5, 9, 15, 20])

    # Need fine-grained labels for purity heatmaps — requires labeled test set
    labeled_csv = cfg["paths"].get("test_set_labeled", None)

    print("Extracting embeddings ...")
    X, y, fine_labels = extract_embeddings(cfg, split="train+val")

    # ---- Search on ALL samples ----
    print(f"\nSearching K in [{k_min}, {k_max}] on ALL embeddings (N={len(X)}) ...")
    results_all = search_k(X, k_range=range(k_min, k_max + 1),
                           n_init=n_init, use_gmm=use_gmm)
    print_summary(results_all)
    plot_k_search(results_all,
                  title_suffix="(all classes)",
                  save_path=f"{output_path}/k_search_all.png",
                  use_gmm=use_gmm)

    print(f"\nGenerating silhouette diagrams for K={sil_ks} (all classes) ...")
    plot_silhouette_diagrams(X, k_values=sil_ks, n_init=n_init,
                             save_path=f"{output_path}/silhouette_all",
                             title_suffix="(all classes)")

    if fine_labels is not None:
        print(f"\nGenerating purity heatmaps for K={sil_ks} (all classes) ...")
        generate_purity_heatmaps(X, fine_labels, k_values=sil_ks, n_init=n_init,
                                 save_path=f"{output_path}/purity_all",
                                 title_suffix="(all classes)")

    # ---- Search on NON-METEORS only ----
    X_nm = filter_non_meteors(X, y)
    fine_nm = fine_labels[y == 0] if fine_labels is not None else None

    print(f"\nSearching K in [{k_min}, {k_max}] on NON-METEOR embeddings (N={len(X_nm)}) ...")
    results_nm = search_k(X_nm, k_range=range(k_min, k_max + 1),
                          n_init=n_init, use_gmm=use_gmm)
    print_summary(results_nm)
    plot_k_search(results_nm,
                  title_suffix="(non-meteors only)",
                  save_path=f"{output_path}/k_search_non_meteors.png",
                  use_gmm=use_gmm)

    print(f"\nGenerating silhouette diagrams for K={sil_ks} (non-meteors only) ...")
    plot_silhouette_diagrams(X_nm, k_values=sil_ks, n_init=n_init,
                             save_path=f"{output_path}/silhouette_non_meteors",
                             title_suffix="(non-meteors only)")

    if fine_nm is not None:
        print(f"\nGenerating purity heatmaps for K={sil_ks} (non-meteors only) ...")
        generate_purity_heatmaps(X_nm, fine_nm, k_values=sil_ks, n_init=n_init,
                                 save_path=f"{output_path}/purity_non_meteors",
                                 title_suffix="(non-meteors only)")

    return results_all, results_nm
