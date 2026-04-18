import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

import models.ssl_model as encoder
import models.classifiers as classifiers
import utils.plotting as plots
import transformations.transform as transform
import data.datasets as datasets
from data.dataloaders import get_ssl_loader, get_two_view_loader

def plot_model_results(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = cfg['paths']['output_dir']
    batch_size = cfg['batch_size']
    VERSION = cfg['experiment_version']
    
    # ----------------------------------
    # Load model and loaders
    # ----------------------------------
    ssl_model = encoder.get_model(cfg['model_path'])   # loads the model with final architecture

    train_set, val_set, test_set = datasets.get_dataset_split(full_dataset_csv_path=cfg['paths']['full_dataset'], output_path=cfg['paths']['datasets_dir'])
    train_set, train_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=train_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION)
    val_set, val_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=val_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION)
    test_set, test_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=test_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION)
    print(f"Dataset: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    _, two_view_loader = get_two_view_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=train_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION)
    
    N = len(val_loader)

    # -----------------------------------------------
    # Extract features for plots on validation set
    # -----------------------------------------------
    X_backbone, X_projection, y_true = encoder.get_encoding_and_projection(
        model=ssl_model,
        dataloader=val_loader,
        device=device
    )
    X_projection_head_i, X_projection_head_j = encoder.get_two_augmentations_projection(
        model=ssl_model,
        dataloader=two_view_loader,
        device=device
    )    
    X_backbone_norm = X_backbone / np.linalg.norm(X_backbone, axis=1, keepdims=True)


    # -------------------------------------------
    # Obtain classification results
    # -------------------------------------------
    # Get classifier
    clf = encoder.get_model(cfg['classifier_model_path'])
    y_pred, y_score = classifiers.predict(clf, X_backbone, y_true)


    # -------------------------------------------
    # Plots
    # -------------------------------------------
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=["meteor", "unknown"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["meteor", "non-meteor"])
    disp.figure_.savefig(f"{output_path}/confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label="meteor")
    roc_auc = auc(fpr, tpr)
    fig = plots.plot_roc_curve(fpr, tpr, roc_auc)
    fig.savefig(f"{output_path}/roc_curve.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # t-SNE embeddings plot
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto')
    Z = tsne.fit_transform(X_backbone_norm) # to avoid distortions
    fig = plots.plot_tsne_3d(Z, labels=y_true)
    fig.savefig("umap_3d.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Alignment and Uniformity Diagnostic with projection head embeddings
    distances = encoder.compute_augmentations_distance(X_projection_head_j, X_projection_head_j)
    fig = plots.plot_alignment_hist(distances)
    fig.savefig(f"{output_path}/umap_3d.png", dpi=300, bbox_inches='tight')
    plt.close(f"{output_path}/alignment_distribution.png", dpi=300)

    # Compute angles, and KDE
    X_proj_2d = transform.project_2d_hypersphere(X_projection_head_i)
    x, y = X_proj_2d[:, 0], X_proj_2d[:,1]
    angles = np.arctan2(y, x)

    kde_2d = gaussian_kde(np.stack([x, y])) # compute KDE approximation with 2d embeddings
    angle_kde = gaussian_kde(angles)        # compute 1D KDE for the Angular Density

    fig = plots.plot_uniformity_plots(kde_2d, angle_kde, count=N)
    fig.savefig(f"{output_path}/uniformity_plots.png", dpi=300)
    plt.close()
