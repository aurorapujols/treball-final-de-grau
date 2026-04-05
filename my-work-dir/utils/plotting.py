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
    out_path = os.path.join(save_path, f"triplet_v{version}.png")
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

