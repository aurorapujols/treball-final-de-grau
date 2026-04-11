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
CSV_PATH =  "../../../../data/upftfg26/apujols/processed/dataset_51700.csv"
MODEL_PATH = os.path.join(BASE, "optuna_logs", "ssl_best_model_4.1.pt")
FILENAMES_PATH = os.path.join(BASE, "optuna_logs", "ssl_filenames_ssl_final_model_4.1.npy")
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
        res_net_dim=512,
        projection_dim=256
    )
    state = torch.load(filepath, map_location="cpu", weights_only=True) 
    model.load_state_dict(state) 
    model.eval()
    return model

def encode_image(model, df, augment=None, img_name=None):
    # 1. Load image
    img_path = f"{IMAGES_FOLDER}/{img_name}_CROP_SUMIMG.png"
    img = Image.open(img_path).convert("L")   # ensure grayscale
    
    # 2. Apply base transform → tensor
    x = base_transform(img)              # shape (1,128,128)
    
    # 3. Apply augmentations if provided
    if augment is not None:
        # augment.one_view expects a batch and filenames
        x_aug = augment.one_view(x, img_name, df.loc[df['filename'] == img_name, "bmin"].iloc[0], df.loc[df['filename'] == img_name, "bmax"].iloc[0])
    else:
        x_aug = x
    
    # 4. Add batch dimension
    x_aug = x_aug.unsqueeze(0)
    
    # 5. Encode
    with torch.no_grad():
        h, z = model(x_aug)
    
    return h, z

def collect_features(filenames_path, images_folder, model, use_backbone=True, augment=None, label=None):
    filenames = np.load(filenames_path, allow_pickle=True)
    df = pd.read_csv(CSV_PATH, sep=";")
    feats = []
    count = 0

    for img_name in filenames:
        rows = df.loc[df["filename"] == img_name, "class"]
        if rows.empty:
            continue
        my_class = rows.iloc[0]
        if label is None or my_class == label:
            h, z = encode_image(model=model, df=df, augment=augment, img_name=img_name)
            
            if use_backbone:
                # IMPORTANT: Backbone (h) is not usually L2-normalized by the model.
                # We MUST normalize it here for the Hypersphere metrics to make sense.
                feat = h / torch.norm(h, p=2, dim=1, keepdim=True)
            else:
                feat = z # z is usually already normalized in the model's forward pass
                
            feats.append(feat.squeeze(0).cpu().numpy())
            count += 1

    return np.vstack(feats), count   # shape (N, d)

def project_to_2d(zs):
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(zs)
    # Normalize to unit circle
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    return z2

def project_to_circle(zs):
    """
    Correctly projects high-dim features to the unit circle for visualization.
    1. PCA to 2D
    2. L2 Normalization so all points sit on the circle edge.
    """
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(zs)
    # This is the step that creates the 'ring' or 'circle' shape for the KDE
    z2_norm = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    return z2_norm

def get_alignment_diagnostic(on_backbone=True):
    """
    Computes distances between positive pairs (same image, different augment).
    """
    distances = compute_alignments(filenames=FILENAMES_PATH, on_backbone=on_backbone)
    
    plt.figure(figsize=(8, 5))
    # Using density=True makes it a proper probability distribution like the paper
    plt.hist(distances, bins=30, density=True, color="skyblue", alpha=0.7, edgecolor="black")
    
    mean_val = np.mean(distances)
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
    
    plt.title(f"Alignment: Positive Pair Distances\n({'Backbone' if on_backbone else 'Projection Head'})")
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.xlim(0, 2) # Max distance on unit sphere is 2
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f"{OUTPUT_FOLDER}/alignment_distribution.png", dpi=300)
    plt.close()

def get_uniformity_diagnostic_v2(label=None):
    """
    Recreates the Wang & Isola Figure 3 visual style.
    """
    model = get_model()
    # Collect features without augmentations for uniformity
    feats, count = collect_features(FILENAMES_PATH, IMAGES_FOLDER, model, augment=None, label=label)
    
    # 1. Project to 2D circle
    feats2 = project_to_circle(feats)
    x, y = feats2[:,0], feats2[:,1]
    angles = np.arctan2(y, x)

    # 2. Compute 2D KDE for the 'circular' heatmap
    kde2d = gaussian_kde(np.vstack([x, y]))
    # Grid slightly larger than the circle
    xx, yy = np.mgrid[-1.3:1.3:150j, -1.3:1.3:150j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde2d(grid).reshape(xx.shape)

    # 3. Compute 1D KDE for the Angular Density
    angle_kde = gaussian_kde(angles)
    angle_grid = np.linspace(-np.pi, np.pi, 500)
    angle_density = angle_kde(angle_grid)

    # Visualization
    fig, (ax_circle, ax_angle) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: The Circle Plot
    ax_circle.contourf(xx, yy, zz, levels=20, cmap='Blues')
    # Draw unit circle reference
    unit_circle = plt.Circle((0,0), 1, color='black', fill=False, linestyle='--', alpha=0.5)
    ax_circle.add_artist(unit_circle)
    ax_circle.set_aspect('equal')
    ax_circle.set_xlim(-1.3, 1.3); ax_circle.set_ylim(-1.3, 1.3)
    ax_circle.set_title(f"Uniformity: Feature Distribution (n={count})")

    # Right: The Angle Plot
    ax_angle.fill_between(angle_grid, angle_density, color='blue', alpha=0.3)
    ax_angle.plot(angle_grid, angle_density, color='darkblue', lw=2)
    ax_angle.set_xticks([-np.pi, 0, np.pi])
    ax_angle.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax_angle.set_title("Uniformity: Angular Density")
    ax_angle.set_xlabel("Angle (radians)")

    plt.tight_layout()
    label_str = label if label is not None else "all_classes"
    plt.savefig(f"{OUTPUT_FOLDER}/uniformity_diag_{label_str}.png", dpi=300)
    plt.close()


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
def compute_alignments(filenames, on_backbone=True):
    distances = []
    model = get_model()
    augmentfn = ControlledAugment(use_enhanced=True)
    filenames = np.load(filenames, allow_pickle=True)
    df = pd.read_csv(CSV_PATH, sep=";")
    for img_name in filenames:
        hi, zi = encode_image(model=model, df=df, augment=augmentfn, img_name=img_name)
        hj, zj = encode_image(model=model, df=df, augment=augmentfn, img_name=img_name)
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
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Run the specific diagnostics based on your code logic
    print("Computing Alignment...")
    get_alignment_diagnostic(on_backbone=True)
    
    print("Computing Uniformity for all classes...")
    get_uniformity_diagnostic_v2(label=None)
