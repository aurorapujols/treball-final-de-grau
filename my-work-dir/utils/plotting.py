import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import numpy as np

from scipy.stats import gaussian_kde
from sklearn.metrics import ConfusionMatrixDisplay

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

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["meteor", "non-meteor"]
    )

    disp.plot(ax=ax,cmap="Blues", colorbar=False)

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()

    return fig

def plot_class_kde(yhat, y_test):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define classes and colors
    # Assuming 0: non-meteor, 1: meteor based on your previous logic
    classes = [0, 1]
    colors = ["#4B0082", "#FFD700"]
    labels = ["non-meteor", "meteor"]
    
    for cls, color, label in zip(classes, colors, labels):
        data = yhat[np.where(y_test == cls)]
        if len(data) > 1: # KDE needs at least two points to estimate bandwidth
            kde = gaussian_kde(data)
            x_range = np.linspace(0, 1, 200) # Probabilities are 0 to 1
            ax.plot(x_range, kde(x_range), color=color, lw=2, label=label)
            ax.fill_between(x_range, kde(x_range), color=color, alpha=0.3)
    
    ax.set_title('Distribution of Predicted Probabilities by True Class')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()
    return fig

def plot_tsne_3d(Z, labels):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Map 0/1 → colors
    # If you prefer viridis, keep c=labels and remove this line
    colors = np.where(labels == 1, "#FFD700", "#4B0082")

    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors, s=8, alpha=0.8)

    # Manual legend (3D scatter cannot auto-generate)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='non-meteor',
               markerfacecolor='#4B0082', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='meteor',
               markerfacecolor='#FFD700', markersize=8)
    ]
    ax.legend(handles=legend_elements)

    ax.set_title("3D t-SNE Embeddings")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    plt.tight_layout()
    return fig

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

    fig = plt.figure(figsize=(7, 9))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_circle = fig.add_subplot(gs[0])
    ax_angle = fig.add_subplot(gs[1])
    
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

    return fig

def plot_knn_distance_kde(distances, y_true, k):
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = [0, 1]
    colors = ["#4B0082", "#FFD700"]  # indigo + gold
    names = ["non-meteor", "meteor"]

    x_range = np.linspace(0, np.max(distances) * 1.1, 500)

    for cls, color, name in zip(classes, colors, names):
        cls_dist = distances[y_true == cls]

        if len(cls_dist) > 1:
            kde = gaussian_kde(cls_dist)
            density = kde(x_range)

            ax.plot(x_range, density, color=color, lw=2, label=name)
            ax.fill_between(x_range, density, color=color, alpha=0.25)

            # Median line
            median_dist = np.median(cls_dist)
            ax.axvline(median_dist, color=color, linestyle='--',
                       alpha=0.8, lw=1.5)

    ax.set_title(f"Local Density: {k}-NN Distances by Class")
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    return fig