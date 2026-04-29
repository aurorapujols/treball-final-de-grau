import os
import torch
import joblib
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

import models.ssl_model as encoder
import models.classifiers as classifiers
import utils.plotting as plots
import transformations.transform as transform
import data.datasets as datasets
from data.dataloaders import get_ssl_loader, get_two_view_loader
from models.modules import get_vgg16_embedded_images

def plot_model_results(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_cfg = cfg.get('plot_options', {})

    output_path = cfg['paths']['output_dir']
    batch_size = cfg['batch_size']
    VERSION = cfg['experiment_version']
    
    # ----------------------------------
    # Load model and loaders
    # ----------------------------------
    ssl_model = encoder.get_model(cfg['ssl_model_path'])   # loads the model with final architecture

    train_set, val_set, test_set = datasets.get_dataset_split(full_dataset_csv_path=cfg['paths']['full_dataset'], output_path=cfg['paths']['datasets_dir'])
    train_set_l, train_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=train_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION,
        shuffle=False)
    # val_set_l, val_loader = get_ssl_loader(
    #     data_root=cfg['paths']['data_root'], 
    #     dataframe=val_set,
    #     batch_size=batch_size,
    #     transform=transform.base_transform,
    #     version=VERSION,
    #     shuffle=False)
    test_set_l, test_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=test_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION,
        shuffle=False)
    print(f"Dataset: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    _, two_view_loader = get_two_view_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=test_set,
        batch_size=batch_size,
        transform=transform.base_transform,
        version=VERSION)
    
    N = len(test_set)

    # -----------------------------------------------
    # Extract features for plots on validation set
    # -----------------------------------------------
    X_backbone, X_projection, y_true = encoder.get_encoding_and_projection(
        model=ssl_model,
        dataloader=test_loader,
        device=device
    )
    X_projection_head_i, X_projection_head_j = encoder.get_two_augmentations_projection(
        model=ssl_model,
        dataloader=two_view_loader,
        device=device
    )    
    X_backbone_norm = X_backbone / np.linalg.norm(X_backbone, axis=1, keepdims=True)

    label_map = {"unknown": 0, "meteor": 1}
    y_true = np.array([label_map[b] for b in y_true])

    # -------------------------------------------
    # Obtain classification results
    # -------------------------------------------
    # Get classifier
    clf = joblib.load(cfg['classifier_model_path'])
    y_pred, y_probs = classifiers.predict(clf, X_backbone, threshold=0.5)


    # -------------------------------------------
    # Plots
    # -------------------------------------------
    # VGG16 embeddings t-SNE
    if plot_cfg.get('vgg16_tsne_plot', False):
        print("\nWorking on VGG16 t-SNE plot...")
        X, y = get_vgg16_embedded_images(cfg['paths']['data_root'], cfg['paths']['test_set'])
        y = np.array([label_map[b] for b in y], dtype=np.int64)
        tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='pca')
        Z = tsne.fit_transform(X) # to avoid distortions
        fig = plots.plot_tsne_3d(Z, labels=y)
        fig.savefig(f"{output_path}/tsne_3d_{VERSION}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Confusion Matrix Plot
    if plot_cfg.get('conf_matrix', False):
        print("\nWorking on confusion matrix...")
        print(classification_report(y_true, y_pred, target_names=["non-meteor", "meteor"]))
        cm = confusion_matrix(y_true, y_pred, labels=[1,0])
        fig = plots.plot_confusion_matrix(cm)
        fig.savefig(f"{output_path}/confusion_matrix_{VERSION}.png", dpi=300)
        plt.close(fig)

    # Plot Class KDE
    if plot_cfg.get('class_kde', False):
        print("\nWorking on classes kde...")
        fig = plots.plot_class_kde(y_pred, y_true)  # in validation set
        fig.savefig(f"{output_path}/class_kde_{VERSION}.png", dpi=300)
        plt.close()

    # ROC Curve
    if plot_cfg.get('roc_plot', False):
        print("\nWorking on roc curve...")
        fpr, tpr, thresholds = roc_curve(y_true, y_probs[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        fig = plots.plot_roc_curve(fpr, tpr, roc_auc)
        fig.savefig(f"{output_path}/roc_curve_{VERSION}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # t-SNE embeddings plot
    if plot_cfg.get('tsne_plot', False):
        print("\nWorking on t-SNE plot...")
        tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto')
        Z = tsne.fit_transform(X_backbone_norm) # to avoid distortions
        fig = plots.plot_tsne_3d(Z, labels=y_true)
        fig.savefig(f"{output_path}/tsne_3d_{VERSION}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Alignment and Uniformity Diagnostic with projection head embeddings
    if plot_cfg.get('alignment_plot', False):
        print("\nWorking on alignment plot...")
        distances = encoder.compute_augmentations_distance(X_projection_head_i, X_projection_head_j)
        fig = plots.plot_alignment_hist(distances)
        fig.savefig(f"{output_path}/alignment_distribution_{VERSION}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Compute angles, and KDE
    if plot_cfg.get('uniformity_plot', False):
        print("\nWorking on uniformity plot...")
        X_proj_2d = transform.project_2d_hypersphere(X_projection)
        x, y = X_proj_2d[:, 0], X_proj_2d[:,1]
        angles = np.arctan2(y, x)

        kde_2d = gaussian_kde(np.stack([x, y])) # compute KDE approximation with 2d embeddings
        angle_kde = gaussian_kde(angles)        # compute 1D KDE for the Angular Density

        fig = plots.plot_uniformity_plots(kde_2d, angle_kde, count=N)
        fig.savefig(f"{output_path}/uniformity_plots_{VERSION}.png", dpi=300)
        plt.close()

    # Histogram of k-NN
    if plot_cfg.get('knn_kde', False):
        print("\nWorking on k-NN mean distances KDE plot...")
        k = 5  # Common choice for local density
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_backbone_norm)
        distances, indices = nbrs.kneighbors(X_backbone_norm)
        
        # We take the distance to the kth neighbor (ignoring the 0th which is the point itself)
        knn_dist = distances[:, k]
        
        fig = plots.plot_knn_distance_kde(knn_dist, y_true, k)
        fig.savefig(f"{output_path}/knn_distance_kde_{VERSION}.png", dpi=300)
        plt.close()

    # -----------------------------------------------------------------
    # Move missclassifications in different folders to visualize them
    # -----------------------------------------------------------------
    results_df = test_set.copy()  # avoid modifying original
    results_df["y_true"] = y_true
    results_df["y_pred"] = y_pred

    if hasattr(clf, "classes_"):
        idx_meteor = np.where(clf.classes_ == 1)[0][0]
        idx_non_meteor = np.where(clf.classes_ == 0)[0][0]
    else:
        idx_meteor = 1
        idx_non_meteor = 0
    p_meteor = y_probs[:, idx_meteor]
    p_non_meteor = y_probs[:, idx_non_meteor]
    results_df["meteor_prob"] = p_meteor
    results_df["non-meteor_prob"] = p_non_meteor

    results_df.to_csv(f"{output_path}/classification_results_val.csv", sep=";")

    # Misclassifications
    false_positives  = results_df[(results_df.y_true == 0) & (results_df.y_pred == 1)]
    false_negatives  = results_df[(results_df.y_true == 1) & (results_df.y_pred == 0)]

    fp_filenames = false_positives["filename"].tolist()
    fn_filenames = false_negatives["filename"].tolist()

    data_root = cfg['paths']['data_root']

    def clear_folder(path):
            if os.path.exists(path):
                shutil.rmtree(path)   # delete folder and all contents
            os.makedirs(path, exist_ok=True)  # recreate empty folder

    dest_folder = f"{cfg['paths']['fp_dest']}_{VERSION}"
    clear_folder(dest_folder)    
    for filename in fp_filenames:
        os.makedirs(dest_folder, exist_ok=True)
        src_path = os.path.join(data_root, f"{filename}_CROP_SUMIMG.png")
        dst_path = os.path.join(dest_folder, f"{filename}.png")
        shutil.copy(src_path, dst_path)
    
    dest_folder = f"{cfg['paths']['fn_dest']}_{VERSION}"
    clear_folder(dest_folder)    
    for filename in fn_filenames:
        os.makedirs(dest_folder, exist_ok=True)
        src_path = os.path.join(data_root, f"{filename}_CROP_SUMIMG.png")
        dst_path = os.path.join(dest_folder, f"{filename}.png")
        shutil.copy(src_path, dst_path)