import pandas as pd
import matplotlib.pyplot as plt
import torch 
import sys, os
import numpy as np 
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 
sys.path.append(ROOT)
from models.ssl_model import SSLResNet
from transformations.augment import ControlledAugment
from transformations.transform import base_transform
from evaluation.metrics import alignment, uniformity

BASE = os.path.dirname(__file__)
CSV_PATH =  "../../../../data/upftfg26/apujols/processed/dataset_36164.csv"
MODEL_PATH = os.path.join(BASE, "optuna_logs", "ssl_model_ssl_optuna_trial_1.25.pt")
FILENAMES_PATH = os.path.join(BASE, "optuna_logs", "ssl_filenames_ssl_optuna_trial_1.25.npy")
IMAGES_FOLDER = "../../../../data/upftfg26/apujols/processed/original"
OUTPUT_FOLDER = os.path.join(BASE, "visualizations")
best_params = {'backbone_dim': 2048, 'hidden_dim': 1792, 'projection_dim': 512, 'lr': 0.00075326638783747, 'temperature': 0.32793068208441767, 'batch_size': 64}

# ---------------------------------------
# Helper functions
# ---------------------------------------
def get_df_from_csv(filepath, sep=";"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath, sep=sep)
    return None

def get_model(filepath=MODEL_PATH):
    model = SSLResNet(
        backbone_dim=best_params["backbone_dim"],
        hidden_dim=best_params["hidden_dim"],
        projection_dim=best_params["projection_dim"]
    )
    state = torch.load(filepath, map_location="cpu", weights_only=True) 
    model.load_state_dict(state) 
    model.eval()
    return model

def encode_image(model, path, augment=None):
    # 1. Load image
    img = Image.open(path).convert("L")   # ensure grayscale
    
    # 2. Apply base transform → tensor
    x = base_transform(img)              # shape (1,128,128)
    
    # 3. Apply augmentations if provided
    if augment is not None:
        # augment.one_view expects a batch and filenames
        x_aug = augment.one_view(x, os.path.basename(path))
    else:
        x_aug = x
    
    # 4. Add batch dimension
    x_aug = x_aug.unsqueeze(0)
    
    # 5. Encode
    with torch.no_grad():
        h, z = model(x_aug)
    
    return h, z

def collect_features(filenames_path, images_folder, model, augment=None, label=None):
    filenames = np.load(filenames_path, allow_pickle=True)
    df = pd.read_csv(CSV_PATH, sep=";")
    zs = []
    count = 0

    for img_name in filenames:
        rows = df.loc[df["filename"] == img_name, "class"]
        if rows.empty:
            continue
        my_class = rows.iloc[0]
        if label is None or my_class == label:
            img_path = os.path.join(images_folder, f"{img_name}_CROP_SUMIMG.png")
            _, z = encode_image(model=model, path=img_path, augment=augment)
            zs.append(z.squeeze(0).cpu().numpy())
            count += 1

    return np.vstack(zs), count   # shape (N, d)

def project_to_2d(zs):
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(zs)
    # Normalize to unit circle
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    return z2

# ---------------------------------------
# 1. Optuna trials' results
# ---------------------------------------
def get_trials_metrics(trials=50):
    filepath = str(os.path.join(BASE, "optuna_logs", "ssl_history_ssl_optuna_trial_1."))

    alignments = []
    uniformities = []
    accuracies = []

    for i in range(trials):
        df = get_df_from_csv(f"{filepath}{i}.csv")
        if df is not None:
            alignments.append(df.iloc[-1]["alignment"])
            uniformities.append(df.iloc[-1]["uniformity"])
            accuracies.append(df.iloc[-1]["accuracy"])
        # else:
        #     print(f"No results for trial {i}")
    
    plt.figure(figsize=(10,7))
    scatter = plt.scatter(
        uniformities,
        alignments,
        c=accuracies,
        cmap="plasma",
        s=80,
        edgecolor="black"
    )

    plt.xlim(-4, -3) 
    plt.ylim(0, 0.4)

    plt.xlabel("Uniformity") 
    plt.ylabel("Alignment") 
    plt.title("Hyperparameter Trials: Alignment vs Uniformity (colored by Accuracy)") 
    cbar = plt.colorbar(scatter) 
    cbar.set_label("Accuracy") 
    plt.grid(True, linestyle="--", alpha=0.4) 
    plt.tight_layout() 
    plt.savefig(f"{OUTPUT_FOLDER}/trials_metrics.png")
    plt.close()

# ---------------------------------------
# 2. Alignment
# ---------------------------------------
def compute_alignments(filenames, on_backbone=False):
    distances = []
    model = get_model()
    augmentfn = ControlledAugment()
    filenames = np.load(filenames, allow_pickle=True)
    for img_name in filenames:
        hi, zi = encode_image(model=model, path=f"{IMAGES_FOLDER}/{img_name}_CROP_SUMIMG.png", augment=augmentfn)
        hj, zj = encode_image(model=model, path=f"{IMAGES_FOLDER}/{img_name}_CROP_SUMIMG.png", augment=augmentfn)
        xi_norm = zi
        xj_norm = zj
        if on_backbone:
            xi_norm = hi / hi.norm(dim=1, keepdim=True)
            xj_norm = hj / hj.norm(dim=1, keepdim=True)
        
        # ℓ₂ distance between normalized projection features 
        d = torch.norm(xi_norm - xj_norm, p=2).item() 
        distances.append(d)

    return np.array(distances)

def plot_alignment_histogram(distances, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.hist(distances, bins=10, color="skyblue", edgecolor="black")
    mean_val = distances.mean()

    # Vertical mean line
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.4f}")

    plt.xlabel("ℓ₂ Distance")
    plt.ylabel("Counts")
    plt.title("Alignment: Positive Pair Feature Distances")
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

def get_alignment_histogram():
    distances = compute_alignments(filenames=FILENAMES_PATH)
    plot_alignment_histogram(distances=distances, save_path=f"{OUTPUT_FOLDER}/align_histogram.png")


# ---------------------------------------
# 3. Uniformity
# ---------------------------------------
def plot_gaussian_kde(z2, save_path=None):
    x, y = z2[:,0], z2[:,1]
    kde = gaussian_kde(np.vstack([x, y]))

    # Grid for density
    xmin = ymin = -1.2
    xmax = ymax = 1.2
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.imshow(zz.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis')
    plt.scatter(x, y, s=2, color='white', alpha=0.5)
    plt.title("Uniformity: Gaussian KDE in ℝ²")
    plt.axis('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_angle_kde(z2, save_path=None):
    x, y = z2[:,0], z2[:,1]
    angles = np.arctan2(y, x)

    plt.figure(figsize=(8,4))
    plt.hist(angles, bins=60, density=True, color="skyblue", edgecolor="black")
    plt.title("Uniformity: Angle Distribution (vMF KDE Approx.)")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Density")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def get_uniformity_plots(model_path=MODEL_PATH, filenames_path=FILENAMES_PATH, label=None):
    model = get_model(model_path)
    augment = None   # uniformity uses *no* augmentations

    # Get all the normalized embeddings (z) for all images in the dataset
    zs = collect_features(
        filenames_path=filenames_path,
        images_folder=IMAGES_FOLDER,
        model=model,
        augment=augment,
        label=label
    )
    # Project in 2D in the hypersphere for visualization
    z2 = project_to_2d(zs)

    plot_gaussian_kde(z2, save_path=f"{OUTPUT_FOLDER}/uniformity_gaussian.png")
    plot_angle_kde(z2, save_path=f"{OUTPUT_FOLDER}/uniformity_angles.png")

def get_uniformity_diagnostic(model_path=MODEL_PATH,
                              filenames_path=FILENAMES_PATH,
                              bins=40,
                              save_path=f"{OUTPUT_FOLDER}/uniformity_diagnostic",
                              label=None):

    model = get_model(model_path)
    augment = None   # uniformity uses *no* augmentations

    # Get normalized embeddings
    zs, count = collect_features(
        filenames_path=filenames_path,
        images_folder=IMAGES_FOLDER,
        model=model,
        augment=augment,
        label=label
    )

    # Project to 2D (PCA → unit circle)
    z2 = project_to_2d(zs)
    x, y = z2[:,0], z2[:,1]

    # Compute angles
    angles = np.arctan2(y, x)

    # KDE for angles
    angle_kde = gaussian_kde(angles)
    angle_grid = np.linspace(-np.pi, np.pi, 400)
    angle_density = angle_kde(angle_grid)

    # 2D KDE
    kde2d = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[-1.2:1.2:200j, -1.2:1.2:200j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde2d(grid).reshape(xx.shape)

    # -------------------------
    # Create 1×2 figure
    # -------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3,1]})

    # -------------------------
    # 1. Angle histogram + KDE
    # -------------------------
    ax[1].hist(angles, bins=bins, density=True, alpha=0.4, color='skyblue')
    ax[1].plot(angle_grid, angle_density, color='darkblue', linewidth=2)
    ax[1].set_title("Angle Distribution (Histogram + KDE)")
    ax[1].set_xlabel("Angle (radians)")
    ax[1].set_ylabel("Density")

    # -------------------------
    # 2. 2D KDE heatmap
    # -------------------------
    ax[0].imshow(
        zz.T, origin='lower',
        extent=[-1.2, 1.2, -1.2, 1.2],
        cmap='viridis'
    )
    ax[0].set_title("2D Gaussian KDE")
    ax[0].set_aspect('equal')
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")

    plt.tight_layout()

    if save_path:
        save_path = save_path + f"_{label if label is not None else 'ALL'}_{count}.png"
        plt.savefig(save_path, dpi=300)
    plt.close()

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":

    # get_trials_metrics()          # 1. To see which models performed best
    get_alignment_histogram()     # 2. To see the histogram of alignment (distances between two positive augmentations)
    # get_uniformity_plots()        # 3. To see the plots for uniformity (angle & PCA in hypersphere)
    # get_uniformity_diagnostic(label=None)    # 4. Uniformity diagnostic
