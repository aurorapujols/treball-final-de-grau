import numpy as np
import pandas as pd
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
from data.datasets import get_dataset_split
from data.dataloaders import get_ssl_loader
from transformations.transform import base_transform


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def extract_embeddings(cfg):
    """Extract and L2-normalize backbone embeddings for train and labeled test set.

    Returns
    -------
    X_train     : (N_train, D) normalized train+val embeddings
    y_train_bin : (N_train,) binary labels  0=non-meteor  1=meteor
    X_test      : (N_test,  D) normalized test embeddings
    y_test_num  : (N_test,)  integer-encoded sublabels from the labeled test CSV
    label_map   : dict {str -> int} built from the test set sublabels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERSION    = cfg["experiment_version"]
    batch_size = cfg.get("batch_size", 128)

    ssl_model = encoder.get_model(cfg["ssl_model_path"])
    ssl_model.to(device)
    ssl_model.eval()

    train_set, val_set, _ = get_dataset_split(
        full_dataset_csv_path=cfg["paths"]["full_dataset"],
        output_path=cfg["paths"]["datasets_dir"]
    )
    train_set = pd.concat([train_set, val_set], ignore_index=True)

    test_set_labeled = pd.read_csv(cfg["paths"]["test_set_labeled"], sep=";")

    # if cfg.get("use_only_non_meteors", False):
    #     train_set        = train_set[train_set["class"] != "meteor"].reset_index(drop=True)
    #     test_set_labeled = test_set_labeled[test_set_labeled["class"] != "meteor"].reset_index(drop=True)
    #     print("Removed all meteor samples from train/test.")

    print(f"Train: {len(train_set)} | Test: {len(test_set_labeled)}")

    _, train_loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        dataframe=train_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION,
        shuffle=False
    )

    _, test_loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        dataframe=test_set_labeled,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION,
        shuffle=False
    )

    X_train, _, y_train_raw = encoder.get_encoding_and_projection(
        model=ssl_model, dataloader=train_loader, device=device
    )
    X_test, _, y_test_raw = encoder.get_encoding_and_projection(
        model=ssl_model, dataloader=test_loader, device=device
    )

    # Binary labels for train (used to filter non-meteors)
    y_train_bin = np.array(
        [1 if lbl == "meteor" else 0 for lbl in y_train_raw], dtype=np.int64
    )

    # Integer sublabels for test (used for purity heatmaps)
    label_map = {}
    for lbl in sorted(set(y_test_raw)):
        label_map[lbl] = len(label_map)
    y_test_num = np.array([label_map[lbl] for lbl in y_test_raw], dtype=np.int64)

    X_train = normalize(X_train, norm="l2")
    X_test  = normalize(X_test,  norm="l2")

    return X_train, y_train_bin, X_test, y_test_num, label_map


def filter_non_meteors(X, y):
    """Return only the non-meteor embeddings."""
    return X[y == 0]


# ----------------------------------------------------------------
# K search
# ----------------------------------------------------------------

def search_k(X, k_range=range(2, 31), n_init=20, random_state=42, use_gmm=False):
    """Fit K-Means (and optionally GMM) for each K and record cluster quality metrics."""
    k_values    = list(k_range)
    inertias    = []
    silhouettes = []
    davies      = []
    bic_scores  = []
    aic_scores  = []

    for k in k_values:
        print(f"  K={k:3d} ...", end=" ", flush=True)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

        sil = silhouette_score(X, labels, sample_size=5000, random_state=random_state)
        db  = davies_bouldin_score(X, labels)
        silhouettes.append(sil)
        davies.append(db)

        if use_gmm:
            gmm = GaussianMixture(
                n_components=k, covariance_type="diag", n_init=3, random_state=random_state
            )
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            print(f"sil={sil:.4f}  db={db:.4f}  bic={gmm.bic(X):.1f}")
        else:
            print(f"sil={sil:.4f}  db={db:.4f}  inertia={km.inertia_:.1f}")

    results = {
        "k": k_values, "inertia": inertias,
        "silhouette": silhouettes, "davies_bouldin": davies,
    }
    if use_gmm:
        results["bic"] = bic_scores
        results["aic"] = aic_scores

    return results


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

def plot_k_search(results, title_suffix="", save_path=None, use_gmm=False):
    k       = results["k"]
    sil     = results["silhouette"]
    db      = results["davies_bouldin"]
    inertia = results["inertia"]

    best_k_sil = k[np.argmax(sil)]
    best_k_db  = k[np.argmin(db)]

    n_plots = 4 if use_gmm else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle(f"K-Means Cluster Search  {title_suffix}", fontsize=13)

    axes[0].plot(k, inertia, "o-", color="steelblue")
    axes[0].set_title("Elbow (Inertia)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia"); axes[0].grid(alpha=0.3)

    axes[1].plot(k, sil, "o-", color="darkorange")
    axes[1].axvline(best_k_sil, linestyle="--", color="darkorange", alpha=0.6,
                    label=f"best K={best_k_sil}")
    axes[1].set_title("Silhouette Score  (↑ better)")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].plot(k, db, "o-", color="seagreen")
    axes[2].axvline(best_k_db, linestyle="--", color="seagreen", alpha=0.6,
                    label=f"best K={best_k_db}")
    axes[2].set_title("Davies-Bouldin Index  (↓ better)")
    axes[2].set_xlabel("K"); axes[2].set_ylabel("DB Index")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    if use_gmm and "bic" in results:
        best_k_bic = k[np.argmin(results["bic"])]
        axes[3].plot(k, results["bic"], "o-", color="purple", label="BIC")
        axes[3].plot(k, results["aic"], "s--", color="crimson", label="AIC")
        axes[3].axvline(best_k_bic, linestyle=":", color="purple", alpha=0.6,
                        label=f"best BIC K={best_k_bic}")
        axes[3].set_title("GMM BIC / AIC  (↓ better)")
        axes[3].set_xlabel("K"); axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)

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

    print("\n========== K Search Summary ==========")
    print(f"  Best K by Silhouette Score : K={k[np.argmax(sil)]}  (score={max(sil):.4f})")
    print(f"  Best K by Davies-Bouldin   : K={k[np.argmin(db)]}  (score={min(db):.4f})")
    if "bic" in results:
        print(f"  Best K by GMM BIC          : K={k[np.argmin(results['bic'])]}")
    print("======================================\n")


# ----------------------------------------------------------------
# Silhouette diagrams
# ----------------------------------------------------------------

def plot_silhouette_diagrams(X, k_values, n_init=20, random_state=42,
                             save_path=None, title_suffix=""):
    from sklearn.decomposition import PCA

    print("  Computing 2D PCA projection for silhouette diagrams...")
    pca  = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    for k in k_values:
        print(f"  Silhouette diagram K={k} ...", end=" ", flush=True)

        km     = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)

        sil_vals = silhouette_samples(X, labels)
        avg_sil  = sil_vals.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Silhouette Diagram  K={k}  {title_suffix}\nMean silhouette = {avg_sil:.4f}",
            fontsize=12
        )

        colors  = cm.tab20(np.linspace(0, 1, k))
        y_lower = 10
        for i in range(k):
            cluster_sil = np.sort(sil_vals[labels == i])
            size    = len(cluster_sil)
            y_upper = y_lower + size
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                              facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
            ax1.text(-0.05, y_lower + size / 2, str(i), fontsize=7)
            y_lower = y_upper + 10

        ax1.axvline(avg_sil, color="red", linestyle="--", lw=1.5, label=f"mean={avg_sil:.3f}")
        ax1.set_xlabel("Silhouette coefficient"); ax1.set_ylabel("Cluster")
        ax1.set_title("Per-point silhouette values")
        ax1.set_xlim([-0.1, 1.0]); ax1.set_yticks([]); ax1.legend(fontsize=9)

        for i in range(k):
            mask = labels == i
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], color=colors[i], s=4, alpha=0.5)
            cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
            ax2.text(cx, cy, str(i), fontsize=8, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.2", fc="white", alpha=0.6))

        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
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

def compute_purity_matrix(cluster_labels, fine_labels, label_map):
    unique_clusters = np.unique(cluster_labels)
    unique_classes  = np.unique(fine_labels)

    purity = np.zeros((len(unique_clusters), len(unique_classes)), dtype=float)
    for i, c in enumerate(unique_clusters):
        mask  = cluster_labels == c
        total = mask.sum()
        for j, cls in enumerate(unique_classes):
            purity[i, j] = (fine_labels[mask] == cls).sum() / total * 100

    inv_map       = {v: k for k, v in label_map.items()}
    class_names   = [inv_map[c] for c in unique_classes]
    cluster_names = [f"C{c}" for c in unique_clusters]

    return purity, cluster_names, class_names


def plot_purity_heatmap(purity_matrix, cluster_names, class_names,
                        title="Cluster Purity Heatmap", save_path=None):
    dominant        = np.argmax(purity_matrix, axis=1)
    sort_idx        = np.argsort(dominant, kind="stable")
    purity_sorted   = purity_matrix[sort_idx]
    clusters_sorted = [cluster_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8),
                                    max(8,  len(cluster_names) * 0.45)))
    im   = ax.imshow(purity_sorted, cmap="viridis", vmin=0, vmax=100, aspect="auto")
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

    for i in range(purity_sorted.shape[0]):
        for j in range(purity_sorted.shape[1]):
            val = purity_sorted[i, j]
            if val > 5:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if val > 50 else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


def generate_purity_heatmaps(X_train, X_test, fine_labels_test, label_map,
                              k_values, n_init=20, random_state=42,
                              save_path=None, title_suffix=""):
    """Fit KMeans on X_train, assign X_test to clusters, plot purity using test sublabels."""
    for k in k_values:
        print(f"  Purity heatmap K={k} ...", end=" ", flush=True)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X_train)
        cluster_labels = km.predict(X_test)

        purity, cluster_names, class_names = compute_purity_matrix(
            cluster_labels, fine_labels_test, label_map
        )
        sp = f"{save_path}_K{k}.png" if save_path else None
        plot_purity_heatmap(
            purity, cluster_names, class_names,
            title=f"Cluster Purity Heatmap  K={k}  {title_suffix}",
            save_path=sp
        )
        if not sp:
            print()


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

    # ----------------------------------------------------------------
    # 1. Extract embeddings — train+val for fitting, test for purity
    # ----------------------------------------------------------------
    print("Extracting embeddings ...")
    X_train, y_train_bin, X_test, y_test_num, label_map = extract_embeddings(cfg)

    # ----------------------------------------------------------------
    # 2. K search + silhouette + purity on ALL train samples
    # ----------------------------------------------------------------
    print(f"\nSearching K in [{k_min}, {k_max}] on ALL train embeddings (N={len(X_train)}) ...")
    results_all = search_k(X_train, k_range=range(k_min, k_max + 1),
                           n_init=n_init, use_gmm=use_gmm)
    print_summary(results_all)
    plot_k_search(results_all, title_suffix="(all classes)",
                  save_path=f"{output_path}/k_search_all.png", use_gmm=use_gmm)

    print(f"\nGenerating silhouette diagrams for K={sil_ks} (all classes) ...")
    plot_silhouette_diagrams(X_train, k_values=sil_ks, n_init=n_init,
                             save_path=f"{output_path}/silhouette_all",
                             title_suffix="(all classes)")

    print(f"\nGenerating purity heatmaps for K={sil_ks} (all classes, test sublabels) ...")
    generate_purity_heatmaps(
        X_train=X_train, X_test=X_test,
        fine_labels_test=y_test_num, label_map=label_map,
        k_values=sil_ks, n_init=n_init,
        save_path=f"{output_path}/purity_all",
        title_suffix="(all classes)"
    )

    # ----------------------------------------------------------------
    # 3. K search + silhouette + purity on NON-METEOR samples only
    # ----------------------------------------------------------------
    X_train_nm = X_train[y_train_bin == 0]

    # Filter test set to non-meteors.
    # "meteor" may not be in label_map if use_only_non_meteors already removed them.
    meteor_int = label_map.get("meteor", None)
    if meteor_int is not None:
        test_nm_mask = y_test_num != meteor_int
    else:
        test_nm_mask = np.ones(len(y_test_num), dtype=bool)  # already all non-meteor
    X_test_nm     = X_test[test_nm_mask]
    y_test_num_nm = y_test_num[test_nm_mask]

    print(f"\nSearching K in [{k_min}, {k_max}] on NON-METEOR train embeddings (N={len(X_train_nm)}) ...")
    results_nm = search_k(X_train_nm, k_range=range(k_min, k_max + 1),
                          n_init=n_init, use_gmm=use_gmm)
    print_summary(results_nm)
    plot_k_search(results_nm, title_suffix="(non-meteors only)",
                  save_path=f"{output_path}/k_search_non_meteors.png", use_gmm=use_gmm)

    print(f"\nGenerating silhouette diagrams for K={sil_ks} (non-meteors only) ...")
    plot_silhouette_diagrams(X_train_nm, k_values=sil_ks, n_init=n_init,
                             save_path=f"{output_path}/silhouette_non_meteors",
                             title_suffix="(non-meteors only)")

    print(f"\nGenerating purity heatmaps for K={sil_ks} (non-meteors only, test sublabels) ...")
    generate_purity_heatmaps(
        X_train=X_train_nm, X_test=X_test_nm,
        fine_labels_test=y_test_num_nm, label_map=label_map,
        k_values=sil_ks, n_init=n_init,
        save_path=f"{output_path}/purity_non_meteors",
        title_suffix="(non-meteors only)"
    )

    return results_all, results_nm
