import os
import matplotlib.pyplot as plt

import numpy as np

def save_plot_augmentations(img_orig, img_i, img_j, save_path, version):
    """
    Saves a 3-panel debug plot showing:
    - original image
    - augmented view 1
    - augmented view 2
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title("original")

    axes[1].imshow(img_i, cmap="gray")
    axes[1].set_title("aug1")

    axes[2].imshow(img_j, cmap="gray")
    axes[2].set_title("aug2")

    plt.tight_layout()
    out_path = os.path.join(save_path)
    fig.savefig(out_path, dpi=150)
    plt.close()

def plot_embeddings_3d(points, labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")   # no mpl_toolkits import needed

    classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for cls, color in zip(classes, colors):
        idx = labels == cls
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            points[idx, 2],
            c=[color],
            label=cls,
            s=40,
            alpha=0.4
        )

    ax.set_title("VGG16 Embeddings (3D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.savefig("embeddings_3d.png")
    plt.close()

def plot_tsne_3d(Z, labels):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        Z[:, 0], Z[:, 1], Z[:, 2],
        c=labels,
        cmap='viridis',
        s=8
    )

    ax.set_title("3D UMAP Embeddings")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    return sc

def plot_roc_curve(fpr, tpr, roc_auc):

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    return plt

def plot_alignment_hist(distances):
    """
    Computes distances between positive pairs (same image, different augmentation).
    """
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=30, density=True, color="skyblue", alpha=0.7, edgecolor="black")
    mean_val = np.mean(distances)
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
    plt.title(f"Alignment: Positive Pair Distances (on projection head embeddings)")
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.xlim(0, 2) # Max distance on unit sphere is 2
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    return plt

def plot_uniformity_plots(kde_2d, angle_kde, count):

    # Grid slightly larger than the circle
    xx, yy = np.mgrid[-1.3:1.3:150j, -1.3:1.3:150j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde_2d(grid).reshape(xx.shape)

    angle_grid = np.linspace(-np.pi, np.pi, 500)
    angle_density = angle_kde(angle_grid)

    fig, (ax_circle, ax_angle) = plt.subplots(2, 1, figsize=(7, 10))
    
    ax_circle.contourf(xx, yy, zz, levels=20, cmap='Blues')
    unit_circle = plt.Circle((0,0), 1, color='black', fill=False, linestyle='--', alpha=0.5)
    ax_circle.add_artist(unit_circle)
    ax_circle.set_aspect('equal')
    ax_circle.set_xlim(-1.3, 1.3)
    ax_circle.set_ylim(-1.3, 1.3)
    ax_circle.set_title(f"Uniformity: Feature Distribution (n={count})")

    ax_angle.fill_between(angle_grid, angle_density, color='blue', alpha=0.3)
    ax_angle.plot(angle_grid, angle_density, color='darkblue', lw=2)
    ax_angle.set_xticks([-np.pi, 0, np.pi])
    ax_angle.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax_angle.set_title("Uniformity: Angular Density")
    ax_angle.set_xlabel("Angle (radians)")

    plt.tight_layout()
    
    return fig