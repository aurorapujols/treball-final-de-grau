import os
import matplotlib.pyplot as plt

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
    plt.savefig(out_path, dpi=150)
    plt.close()
