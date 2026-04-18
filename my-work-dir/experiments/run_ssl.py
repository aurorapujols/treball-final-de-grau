import os
import time
import torch
import optuna
import numpy as np
import pandas as pd

from data.datasets import get_dataset_split
from data.dataloaders import get_ssl_loader
from transformations.transform import base_transform
from models.ssl_model import SSLResNet
from training.ssl_training import train_ssl, extract_backbone_features
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from evaluation.linear_probe import run_linear_probe



def run_ssl_experiment(cfg, add_version=None, augs_idx=None, trial=None, lr=None, temperature=None, batch_size=None):
    """
    Runs a full SSL experiment:
      1. Load unlabeled dataset
      2. Load labeled dataset for linear probe
      3. Train SSL model
      4. Extract features
      5. Save results
    """
    debug = True
    print("\n========== SSL EXPERIMENT ==========")
    print(f"Experiment: {cfg['experiment_name']}")

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if add_version is not None:
        VERSION = f"{cfg['experiment_version'][:2]}{add_version}"
    else:
        VERSION = cfg['experiment_version']
    print(f"VERSION={VERSION}")

    if lr is None and temperature is None and batch_size is None:
        lr = trial.params["lr"] if trial else cfg["training"]["learning_rate"]
        temperature = trial.params["temperature"] if trial else cfg["training"]["temperature"]
        batch_size = trial.params["batch_size"] if trial else cfg["training"]["batch_size"]

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    print("\nLoading datasets...")

    train_set, val_set, test_set = get_dataset_split(full_dataset_csv_path=cfg['paths']['full_dataset'], output_path=cfg['paths']['datasets_dir'])
    train_set, train_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=train_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION)
    val_set, val_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=val_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION)
    test_set, test_loader = get_ssl_loader(
        data_root=cfg['paths']['data_root'], 
        dataframe=test_set,
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION)
    print(f"Dataset: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")



    # -------------------------------------------------
    # Initialize model
    # -------------------------------------------------
    print(f"\nInitializing SSL model with {cfg['model']['loss']} loss ...")

    model = SSLResNet(
        res_net_dim=cfg["model"]["res_net_dim"],
        projection_dim=cfg["model"]["projection_dim"]
    ).to(device)
    
    # -------------------------------------------------
    # Train SSL model
    # -------------------------------------------------
    start_time = time.time()
    params = {
        "num_epochs": int(cfg["training"]["num_epochs"]),
        "patience": int(cfg["training"]["patience"]),
        "max_gap": cfg["training"]["max_gap"],
        "cutoff_ratio": float(cfg["training"]["cutoff_ratio"]),
        "learning_rate": lr,
        "temperature": temperature,
        "loss": cfg["model"]["loss"]
        }
    loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
    args = {
        "device": device,
        "version": VERSION,
        "output_path": cfg["paths"]["output_dir"],
        "augs_idx": augs_idx,
        "use_enhanced": bool(cfg["training"]["use_enhanced"]),
        "eval_every": 1
    }
    model, best_acc, best_state, history, stop_epoch = train_ssl(
        model=model,
        params=params,
        loaders=loaders,
        args=args,
        trial=trial
    )

    training_time = time.time() - start_time

    if debug:
        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"ssl_model_{cfg['experiment_name']}_{VERSION}.pt")
        save_checkpoint(model, ckpt_path)

    print("Loading best model (val_accuracy = {:.4f})".format(best_acc))
    model.load_state_dict(best_state)
    
    if debug:
        best_model_path = os.path.join(output_dir, f"ssl_best_model_{VERSION}.pt")
        save_checkpoint(model, best_model_path)


    # -------------------------------------------------
    # Extract features
    # -------------------------------------------------
    print("\nExtracting backbone features...")
    dataset = pd.read_csv(cfg['paths']['full_dataset'], sep=";")
    print(f"Full dataset length: {len(dataset)}")
    dataset, full_loader = get_ssl_loader(data_root=cfg['paths']['data_root'], dataframe=dataset, batch_size=batch_size, transform=base_transform, version=VERSION, shuffle=False)
    features, _, filenames = extract_backbone_features(model, full_loader, device)


    # -------------------------------------------------
    # Save results
    # -------------------------------------------------    
    print("\nSaving results...")

    # Save training history
    history_path = os.path.join(output_dir, f"ssl_history_{cfg['experiment_name']}_{VERSION}.csv")
    history.to_csv(history_path, sep=";", index=False)
    
    if debug:
        # Save features
        feat_path = os.path.join(output_dir, f"ssl_features_{cfg['experiment_name']}_{VERSION}.npy")
        name_path = os.path.join(output_dir, f"ssl_filenames_{cfg['experiment_name']}_{VERSION}.npy")
        np.save(feat_path, features)
        np.save(name_path, filenames)


    # -------------------------------------------------
    # Save the predictions
    # -------------------------------------------------
    train_feats, train_labels, _ = extract_backbone_features(model, train_loader, device)
    val_feats, val_labels, _ = extract_backbone_features(model, val_loader, device)

    clf, lp_val_accuracy, _ = run_linear_probe(train_feats, train_labels, val_feats, val_labels)
    predictions = clf.predict(features)
    df_pred = pd.DataFrame({
        "filename": filenames,
        "linear_pred": predictions
    })


    df_full = pd.read_csv(cfg['paths']['full_dataset'], sep=";")
    df_merged = df_full.merge(df_pred, on="filename", how="left")
    if debug:
        df_merged.to_csv(f"{cfg['paths']['output_dir']}/predictions_model_{VERSION}.csv", sep=";")
        print(f"Saved predictions CSV for clf with accuracy {lp_val_accuracy:.4f}.")


    # ------------------------------------- ------------
    # Summary
    # -------------------------------------------------
    print("\n========== SSL SUMMARY ==========")
    print(f"Experiment:       {cfg['experiment_name']}")
    print(f"Stop epoch:       {stop_epoch}")
    print(f"Final accuracy:   {history['val_accuracy'].iloc[-1]:.4f}")
    print(f"Final loss:       {history['val_loss'].iloc[-1]:.4f}")
    print(f"Training time:    {int(training_time//3600):.2f}h {(training_time%3600)/60:.2f}min")
    if debug: 
        print(f"Features saved:   {feat_path}")
        print(f"Model saved:      {ckpt_path}")
    print("=================================\n")

    return model, features, filenames, history, stop_epoch, df_merged
