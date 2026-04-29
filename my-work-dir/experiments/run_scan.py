import time
import torch
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from data.dataloaders import get_ssl_loader, get_two_view_loader
from data.datasets import get_dataset_split, NeighborsDataset
from transformations.transform import base_transform
from transformations.augment import ControlledAugmentGPU
import models.ssl_model as encoder
from models.scan_model import ClusteringModel
from training.ssl_training import extract_backbone_features
from training.scan_training import train_scan, train_selflabel
from losses.losses import SCANLoss
from utils.plotting import plot_confusion_matrix_heatmap


def get_train_and_val_neighbors(model, train_loader, val_loader, device, k=20):
    model.eval()

    # ---------------------------------------------------------
    # 1. Extract TRAIN features (database)
    # ---------------------------------------------------------
    print("\tExtracting training features...")
    train_features, _, train_fnames = extract_backbone_features(model, train_loader, device)
    train_features = train_features.astype("float32")
    faiss.normalize_L2(train_features)

    # ---------------------------------------------------------
    # 2. Extract VAL features (queries)
    # ---------------------------------------------------------
    print("\tExtracting validation features...")
    val_features, _, val_fnames = extract_backbone_features(model, val_loader, device)
    val_features = val_features.astype("float32")
    faiss.normalize_L2(val_features)

    # ---------------------------------------------------------
    # 3. Build FAISS index on TRAIN features
    # ---------------------------------------------------------
    d = train_features.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(train_features)

    # ---------------------------------------------------------
    # 4. TRAIN → TRAIN neighbors (self‑neighbors included)
    # ---------------------------------------------------------
    print(f"\tFinding {k} neighbors for TRAIN samples...")
    _, train_neighbors = gpu_index.search(train_features, k + 1)
    # remove self-neighbor at index 0
    train_neighbors = train_neighbors[:, 1:]

    # ---------------------------------------------------------
    # 5. VAL → TRAIN neighbors (cross‑dataset)
    # ---------------------------------------------------------
    print(f"\tFinding {k} neighbors for VAL samples...")
    _, val_neighbors = gpu_index.search(val_features, k)

    # ---------------------------------------------------------
    # 6. Return everything
    # ---------------------------------------------------------
    return {
        "train_neighbors": train_neighbors,   # shape [N_train, K]
        "val_neighbors": val_neighbors,       # shape [N_val,   K]
        "train_fnames": train_fnames,
        "val_fnames": val_fnames
    }

def run_scan(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    scan_cfg = cfg['scan']
    VERSION = cfg['experiment_version']
    batch_size = scan_cfg['loader_batch_size']
    
    print("Starting the SCAN algorithm training.")

    print(f"\nLoading datasets...")
    train_set, val_set, test_set = get_dataset_split(full_dataset_csv_path=cfg['paths']['full_dataset'], output_path=cfg['paths']['datasets_dir'])
    train_dataset, train_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=train_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION,
        shuffle=False)
    val_dataset, val_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=val_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION)
    print(f"Dataset: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # -------------------------------------
    # Stage 1: Embeddings
    # -------------------------------------

    print(f"\nStage 1: Load the Backbone embeddings and compute the {scan_cfg['n_neighbours']}-NN nearest neighbours.")
    
    print(f"\tLoading pretrained backbone")
    # 1. Load Pretrained Backbone
    ssl_model = encoder.get_model(scan_cfg['ssl_model_path'])
    ssl_backbone = ssl_model.encoder
    print("\tSSL Model Type: ",type(ssl_model))
    backbone_dict = {'backbone': ssl_backbone, 'dim': int(scan_cfg['backbone_dim'])}
    ssl_model.eval()
    ssl_backbone.eval()
    
    # 2. Initialize SCAN Model
    print(f"\tInitializing Clustering Model")
    clustering_model = ClusteringModel(backbone_dict, nclusters=int(scan_cfg['n_clusters']), nheads=int(scan_cfg['n_heads']))
    clustering_model.to(device)
    
    # 3. Get Neighbors (with no transform)
    print(f"\tComputing k-NN Neighbours")
    knn_results = get_train_and_val_neighbors(
        model=ssl_backbone, 
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        k=int(scan_cfg['n_neighbours']))    
    
    training_neighbours = knn_results['train_neighbors']
    validation_neighbours = knn_results['val_neighbors']


    print(f"\nStage 2: Prepare and train Clustering Head.")

    # 4. Create SCAN Dataloader
    # We use training transforms (augmentations) here
    print(f"\tLoading NeighboursDataset")
    scan_dataset_train = NeighborsDataset(anchor_dataset=train_dataset, neighbor_dataset=train_dataset, neighbor_indices=training_neighbours)
    scan_dataset_val = NeighborsDataset(anchor_dataset=val_dataset, neighbor_dataset=train_dataset, neighbor_indices=validation_neighbours)
    print(f"\tPreparing SCAN Loader")
    scan_loader_train = DataLoader(scan_dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    scan_loader_val = DataLoader(scan_dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"\tSetting up training configurations")
    scan_augmenter = ControlledAugmentGPU(use_enhanced=True).to(device)     # to apply heavy augmentations
    
    # 5. Training Setup
    scan_lossfn = SCANLoss(entropy_weight=float(scan_cfg['entropy_weight'])).to(device)
    scan_optimizer = torch.optim.SGD(clustering_model.parameters(), lr=float(scan_cfg['lr_scan']), momentum=0.9)

    # 6. Training Loop
    print(f"\tTraining loop started")
    start_time = time.time()
    params = {
        "num_epochs": int(scan_cfg['epochs']),
        "loss_fn": scan_lossfn,
        "optimizer": scan_optimizer,
        "loader": scan_loader_train,
        "val_loader": scan_loader_val,
        "augmenter": scan_augmenter
    }
    args = {
        "device": device,
        "update_cluster_head_only": True    # Freeze the backbone
    }
    model_train_stage2, history = train_scan(model=clustering_model, params=params, args=args)
    history.to_csv(f"{cfg['paths']['output_dir']}/scan_clustering_history.csv", sep=";", index=False)

    training_time = time.time() - start_time
    print(f"Training time: {int(training_time//3600)}h {(training_time%3600)/60:.2f}min")
    scan_model_path = f"{cfg['paths']['output_dir']}/scan_model_stage2.pth"
    torch.save({
        "backbone": model_train_stage2.backbone.state_dict(),
        "cluster_head": model_train_stage2.cluster_head.state_dict(),
        "config": scan_cfg
    }, scan_model_path)

    print(f"Saved SCAN Stage 2 model to {scan_model_path}")
    
    
    # 7. Self-Labeling
    print("\n--- Starting Stage 3: Self-Labeling ---")
    
    # 1. Prepare Self-Labeling Dataset
    # We need a dataset that returns (Weak View, Strong View)
    
    # Setup the loader for self-labeling
    print(f"\tPreparing Self-Labeling dataloader")
    train_sl_dataset, train_sl_loader = get_two_view_loader(
        data_root=cfg['paths']['data_root'],
        dataframe=train_set,
        batch_size=batch_size,
        transform=base_transform, # Strong Augmentation inside with ControlledAugmentGPU
        version=cfg['experiment_version'],
        self_labeling=True  # Get only one augmented view
    )
    val_sl_dataset, val_sl_loader = get_two_view_loader(
        data_root=cfg['paths']['data_root'],
        dataframe=val_set,
        batch_size=batch_size,
        transform=base_transform,
        version=cfg['experiment_version'],
        self_labeling=True  
    )

    # 2. Optimizer for both Backbone and Head
    print(f"\tStarted Self-Labeling Training")
    optimizer_sl = torch.optim.SGD(model_train_stage2.parameters(), lr=float(scan_cfg['lr_selflabeling']), momentum=0.9)

    sl_history = []
    for epoch in range(int(scan_cfg['sl_epochs'])):
        metrics = train_selflabel(
            model_train_stage2,
            train_sl_loader,
            val_sl_loader,
            optimizer_sl,
            device,
            epoch,
            threshold=scan_cfg['prototype_threshold']
        )

        sl_history.append(metrics)

        print(f"Epoch {epoch+1} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Sel: {metrics['train_selection_rate']:.3f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Sel: {metrics['val_selection_rate']:.3f}")

    # 3. (Optional) Save to CSV every epoch so you don't lose progress if it crashes
    sl_history_df = pd.DataFrame(sl_history)
    sl_history_df.to_csv(f"{cfg['paths']['output_dir']}/self_labeling_history.csv", sep=";", index=False)
    scan_model_sl_path = f"{cfg['paths']['output_dir']}/scan_model_stage3_selflabel.pth"
    torch.save({
        "backbone": clustering_model.backbone.state_dict(),
        "cluster_head": clustering_model.cluster_head.state_dict(),
        "config": scan_cfg
    }, scan_model_sl_path)

    print(f"Saved SCAN Stage 3 (self-labeling) model to {scan_model_sl_path}")

def get_cluster_prototypes(model, dataloader, device, head_idx=0):
    model.eval()
    all_features = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack based on your MyMeteorDataset
            imgs, _, _, _, _ = batch 
            imgs = imgs.to(device)

            # 1. Extract backbone features
            feats = model.backbone(imgs)
            
            # 2. Get cluster assignments from the head
            logits = model.cluster_head[head_idx](feats)
            preds = torch.argmax(logits, dim=1)

            all_features.append(feats.cpu())
            all_predictions.append(preds.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    # 3. Calculate the mean feature for each cluster
    n_clusters = model.cluster_head[head_idx].out_features
    prototypes = []
    
    for i in range(n_clusters):
        mask = (all_predictions == i)
        if np.sum(mask) > 0:
            cluster_mean = np.mean(all_features[mask], axis=0)
            prototypes.append(cluster_mean)
        else:
            # Handle empty clusters
            prototypes.append(np.zeros(all_features.shape[1]))

    return np.stack(prototypes)

def get_closest_prototype_images(model, dataloader, device, prototypes, data_root, top_k=1):
    model.eval()
    all_features = []
    all_filenames = []

    with torch.no_grad():
        for batch in dataloader:
            imgs, fnames, _, _, _ = batch
            imgs = imgs.to(device)

            feats = model.backbone(imgs)
            feats = feats.cpu().numpy()

            all_features.append(feats)
            all_filenames.extend(fnames)

    all_features = np.concatenate(all_features, axis=0) # [N, D]
    prototypes = prototypes.astype(np.float32)          # [C, D]

    # Compute distances
    feat_norms = np.sum(all_features**2, axis=1, keepdims=True)     # [N,1]
    proto_norms = np.sum(prototypes**2, axis=1, keepdims=True).T    # [1,C]
    dot = all_features @ prototypes.T                               # [N,C]

    dists = feat_norms + proto_norms - 2 * dot                      # [N,C]

    closest = {}
    for c in range(prototypes.shape[0]):
        idx = np.argsort(dists[:, c])[:top_k]
        closest[c] = [(all_filenames[i], float(dists[i,c])) for i in idx]

    return closest

def get_cluster_mapping_flexible(y_true, y_pred, n_clusters, n_classes):
    """
    Hungarian matching that supports n_clusters >= n_classes.
    Multiple clusters can map to the same label.
    """
    # Contingency matrix: rows = clusters, cols = true classes
    contingency = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for pred, true in zip(y_pred, y_true):
        contingency[pred, true] += 1

    # Hungarian on the full rectangular matrix (minimizes cost → negate)
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Primary mapping from Hungarian
    mapping = {r: c for r, c in zip(row_ind, col_ind)}

    # Any cluster not matched by Hungarian → assign to best class greedily
    for cluster_idx in range(n_clusters):
        if cluster_idx not in mapping:
            mapping[cluster_idx] = int(np.argmax(contingency[cluster_idx]))

    return mapping

def evaluate_clustering(model, test_loader, device, class_names, head_idx=0):

    model.eval()
    all_preds = []
    all_labels = []
    all_fnames = []

    # 1. Collect all raw cluster predictions and true labels
    with torch.no_grad():
        for batch in test_loader:
            imgs, fnames, _, _, labels = batch
            imgs = imgs.to(device)
            
            # Get cluster logits from the specific head
            logits = model(imgs, forward_pass='default')[head_idx]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_fnames.extend(fnames)

    y_pred_raw = np.array(all_preds)
    y_true = np.array(all_labels)
    n_clusters = len(class_names)

    # 2. Map clusters to semantic labels
    mapping = get_cluster_mapping_flexible(y_true, y_pred_raw, n_clusters)
    # Convert "Cluster 3" to "Label 1"
    y_pred_mapped = np.array([mapping[p] for p in y_pred_raw])

    # 3. Create Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_mapped)
    
    # 4. Plotting
    fig = plot_confusion_matrix_heatmap(cm, class_names)
    
    # 5. Create results DataFrame for each sample
    results_df = pd.DataFrame({
        'filename': all_fnames,
        'true_label': [class_names[i] for i in y_true],
        'cluster_idx': y_pred_raw,
        'mapped_label': [class_names[mapping[p]] for p in y_pred_raw]
    })

    return results_df, cm, fig

def plot_cluster_similarity_matrix(model, dataloader, device, head_idx=0, figsize=(10,10)):
    model.eval()
    all_features = []
    all_clusters = []

    # -----------------------------------------
    # 1. Extract features + cluster assignments
    # -----------------------------------------
    with torch.no_grad():
        for batch in dataloader:
            imgs, _, _, _, _ = batch
            imgs = imgs.to(device)

            feats = model.backbone(imgs)
            logits = model.cluster_head[head_idx](feats)
            preds = torch.argmax(logits, dim=1)

            all_features.append(feats.cpu().numpy())
            all_clusters.append(preds.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_clusters = np.concatenate(all_clusters, axis=0)

    # -----------------------------------------
    # 2. Compute cosine similarity matrix
    # -----------------------------------------
    sim_matrix = cosine_similarity(all_features)

    # -----------------------------------------
    # 3. Sort by cluster assignment
    # -----------------------------------------
    sort_idx = np.argsort(all_clusters)
    sim_sorted = sim_matrix[sort_idx][:, sort_idx]
    clusters_sorted = all_clusters[sort_idx]

    # -----------------------------------------
    # 4. Plot
    # -----------------------------------------
    plt.figure(figsize=figsize)
    plt.imshow(sim_sorted, cmap='Blues')
    plt.colorbar(label="Cosine similarity")
    plt.title("Cluster-Sorted Similarity Matrix")
    plt.xlabel("Samples (sorted by cluster)")
    plt.ylabel("Samples (sorted by cluster)")

    # -----------------------------------------
    # 5. Draw cluster boundaries
    # -----------------------------------------
    unique_clusters = np.unique(clusters_sorted)
    boundaries = []
    for c in unique_clusters:
        idx = np.where(clusters_sorted == c)[0]
        boundaries.append((idx[0], idx[-1]))

    for start, end in boundaries:
        plt.axhline(start, color='green', linewidth=1)
        plt.axhline(end, color='green', linewidth=1)
        plt.axvline(start, color='green', linewidth=1)
        plt.axvline(end, color='green', linewidth=1)

    plt.show()

    return sim_sorted, clusters_sorted


def evaluate_scan(cfg):
    """
    Full SCAN evaluation pipeline:
    1. Load model checkpoint (Stage 2 or Stage 3)
    2. Load test dataset + loader
    3. Compute cluster prototypes
    4. Compute cluster mapping (Hungarian)
    5. Evaluate clustering (confusion matrix + per-sample results)
    6. Save everything
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_cfg = cfg["scan"]

    # ---------------------------------------------------------
    # 1. LOAD MODEL CHECKPOINT
    # ---------------------------------------------------------
    checkpoint_path = cfg["paths"]["scan_checkpoint"]   # user chooses stage2 or stage3
    print(f"Loading SCAN model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild backbone dict
    ssl_model = encoder.get_model(scan_cfg["ssl_model_path"])
    backbone_dict = {"backbone": ssl_model.encoder, "dim": scan_cfg["backbone_dim"]}

    # Rebuild SCAN model
    model = ClusteringModel(
        backbone_dict,
        nclusters=scan_cfg["n_clusters"],
        nheads=scan_cfg["n_heads"]
    )
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.cluster_head.load_state_dict(checkpoint["cluster_head"])
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 2. LOAD TEST DATASET
    # ---------------------------------------------------------
    print("Loading test dataset...")

    _, _, test_set = get_dataset_split(
        full_dataset_csv_path=cfg["paths"]["full_dataset"],
        output_path=cfg["paths"]["datasets_dir"]
    )

    test_dataset, test_loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        dataframe=test_set,
        batch_size=scan_cfg["loader_batch_size"],
        transform=base_transform,
        version=cfg["experiment_version"],
        shuffle=False
    )

    class_names = sorted(test_set["class"].unique().tolist())

    # ---------------------------------------------------------
    # 3. COMPUTE CLUSTER PROTOTYPES
    # ---------------------------------------------------------
    print("Computing cluster prototypes...")
    prototypes = get_cluster_prototypes(model, test_loader, device)
    print(f"Prototypes: {prototypes}")

    img_prototypes = get_closest_prototype_images(model, test_loader, device, prototypes, cfg['paths']['data_root'], k = 5)
    print(f"Image prototypes: {img_prototypes}")

    np.save(f"{cfg['paths']['output_dir']}/scan_cluster_prototypes.npy", prototypes)
    print("Saved cluster prototypes.")

    # ---------------------------------------------------------
    # 4. EVALUATE CLUSTERING (Hungarian + Confusion Matrix)
    # ---------------------------------------------------------
    print("Evaluating clustering...")
    results_df, cm, fig = evaluate_clustering(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        head_idx=0
    )

    # Save results
    results_path = f"{cfg['paths']['output_dir']}/scan_clustering_results.csv"
    cm_path = f"{cfg['paths']['output_dir']}/scan_confusion_matrix.npy"
    fig.savefig(f"{cfg['paths']['output_dir']}/confusion_matrix_heatmap.png", dpi=300)

    results_df.to_csv(results_path, sep=";", index=False)
    np.save(cm_path, cm)

    print(f"Saved clustering results to {results_path}")
    print(f"Saved confusion matrix to {cm_path}")

    # ---------------------------------------------------------
    # 5. RETURN EVERYTHING
    # ---------------------------------------------------------
    return {
        "model": model,
        "prototypes": prototypes,
        "results_df": results_df,
        "confusion_matrix": cm,
        "class_names": class_names
    }

def evaluate_scan_test(cfg):
    """
    Full evaluation on the VALIDATION set:
      - Cluster prototypes
      - 5 closest images per prototype
      - Flexible cluster→label mapping (n_clusters >= n_labels)
      - Cluster similarity matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_cfg = cfg["scan"]

    # 1. Load the checkpoint dictionary
    checkpoint_path = cfg['models']['scan_stage3']
    print(f"Loading SCAN checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Rebuild the Model Architecture (using the same logic as your training)
    import models.ssl_model as encoder
    from models.scan_model import ClusteringModel
    
    # Rebuild backbone
    ssl_model = encoder.get_model(scan_cfg["ssl_model_path"])
    backbone_dict = {"backbone": ssl_model.encoder, "dim": int(scan_cfg["backbone_dim"])}

    # Rebuild SCAN model
    model = ClusteringModel(
        backbone_dict,
        nclusters=int(scan_cfg["n_clusters"]),
        nheads=int(scan_cfg["n_heads"])
    )

    # 3. Load the weights from the dictionary into the model
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.cluster_head.load_state_dict(checkpoint["cluster_head"])
    
    model.to(device)
    model.eval()

    # Now this line will work
    device = next(model.parameters()).device

    # ----------------------------------------------------------
    # 1. Load labeled test set
    # ----------------------------------------------------------
    test_set_labeled = pd.read_csv(cfg['paths']['test_set_labeled'], sep=";")

    test_dataset, test_loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        dataframe=test_set_labeled,
        batch_size=scan_cfg["loader_batch_size"],
        transform=base_transform,
        version=cfg["experiment_version"],
        shuffle=False
    )

    class_names = sorted(test_set_labeled["class"].unique().tolist())
    n_classes   = len(class_names)
    n_clusters  = int(scan_cfg["n_clusters"])

    # ----------------------------------------------------------
    # 2. Cluster prototypes
    # ----------------------------------------------------------
    print("Computing cluster prototypes on validation set...")
    prototypes = get_cluster_prototypes(model, test_loader, device, head_idx=0)
    np.save(f"{cfg['paths']['output_dir']}/val_cluster_prototypes.npy", prototypes)
    print(f"  Saved prototypes — shape: {prototypes.shape}")

    # ----------------------------------------------------------
    # 3. 5 closest images per prototype
    # ----------------------------------------------------------
    print("Finding 5 closest images per prototype...")
    closest = get_closest_prototype_images(         # top_k, not k
        model=model,
        dataloader=test_loader,
        device=device,
        prototypes=prototypes,
        data_root=cfg["paths"]["data_root"],
        top_k=5                                     # ← fixed param name
    )
    for cluster_id, imgs in closest.items():
        print(f"  Cluster {cluster_id}:")
        for fname, dist in imgs:
            print(f"    {fname}  (dist={dist:.4f})")

    # ----------------------------------------------------------
    # 4. Collect predictions + ground truth for mapping
    # ----------------------------------------------------------
    print("Collecting predictions for cluster→label mapping...")
    model.eval()

    unique_labels = sorted(test_set_labeled["class"].unique().tolist())
    label_map = {name: i for i, name in enumerate(unique_labels)}
    class_names = unique_labels  # This ensures index 0 matches class_names[0]

    n_classes = len(unique_labels)
    n_clusters = int(scan_cfg['n_clusters'])

    print(f"Automatically detected classes: {label_map}")

    all_preds, all_labels, all_fnames = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs, fnames, _, _, labels = batch
            imgs = imgs.to(device)
            logits = model(imgs, forward_pass='default')[0]   # head 0
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            for label in labels:
                if isinstance(label, str):
                    all_labels.append(label_map[label])
                else:
                    all_labels.append(int(label))   # If they are already a list (e.g. strings or manual list)
            all_fnames.extend(fnames)

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # ----------------------------------------------------------
    # 5. Flexible Hungarian mapping
    # ----------------------------------------------------------
    print("Computing cluster→label mapping (flexible, n_clusters >= n_classes)...")
    mapping = get_cluster_mapping_flexible(y_true, y_pred, n_clusters, n_classes)
    print("  Cluster → Label mapping:")
    for cluster_idx, label_idx in sorted(mapping.items()):
        print(f"    Cluster {cluster_idx:2d} → {class_names[label_idx]}")

    y_pred_mapped = np.array([mapping[p] for p in y_pred])

    # ----------------------------------------------------------
    # 6. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred_mapped)
    fig_cm = plot_confusion_matrix_heatmap(cm, class_names)
    fig_cm.savefig(
        f"{cfg['paths']['output_dir']}/val_confusion_matrix_heatmap.png", dpi=300
    )

    # ----------------------------------------------------------
    # 7. Per-sample results CSV
    # ----------------------------------------------------------
    results_df = pd.DataFrame({
        "filename":     all_fnames,
        "true_label":   [class_names[i] for i in y_true],
        "cluster_idx":  y_pred,
        "mapped_label": [class_names[mapping[p]] for p in y_pred],
    })
    results_path = f"{cfg['paths']['output_dir']}/val_clustering_results.csv"
    results_df.to_csv(results_path, sep=";", index=False)
    print(f"  Saved per-sample results → {results_path}")

    # ----------------------------------------------------------
    # 8. Cluster similarity matrix
    # ----------------------------------------------------------
    print("Plotting cluster similarity matrix...")
    sim_sorted, clusters_sorted = plot_cluster_similarity_matrix(
        model=model,
        dataloader=test_loader,
        device=device,
        head_idx=0
    )

    return {
        "prototypes":        prototypes,
        "closest_images":    closest,
        "mapping":           mapping,
        "results_df":        results_df,
        "confusion_matrix":  cm,
        "sim_matrix_sorted": sim_sorted,
        "clusters_sorted":   clusters_sorted,
    }