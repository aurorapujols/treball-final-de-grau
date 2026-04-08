import os
import time
import copy
import optuna
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from losses.losses import ContrastiveLoss
from transformations.augment import ControlledAugment, RandomAffineMeanFill
from evaluation.linear_probe import run_linear_probe
from evaluation.metrics import StreamingMetrics
from utils.plotting import save_plot_augmentations

def extract_backbone_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            # Case 1: SSL loader → (img, label, Bmin, Bmax)
            if len(batch) == 4:
                imgs, lbls, _, _ = batch

            # Case 2: LP loader → (img, label)
            elif len(batch) == 2:
                imgs, lbls = batch

            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            imgs = imgs.to(device)
            h, _ = model(imgs)
            feats.append(h.cpu())
            labels.extend(lbls)

    feats = torch.cat(feats, dim=0).numpy()
    labels = np.array(labels)
    return feats, labels
    
def compute_val_loss(device, model, ssl_val_loader, augmentfn, lossfn):
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for images, fnames, bmins, bmaxs in ssl_val_loader:
            x_i, x_j = augmentfn(images, fnames, bmins, bmaxs)
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            val_total += lossfn(z_i, z_j).item()
            
        ssl_val_loss = val_total / len(ssl_val_loader)
    
    return ssl_val_loss

def train_ssl(model, num_epochs, patience, cutoff_ratio, lr, temperature, ssl_train_loader, ssl_val_loader, lp_train_loader, lp_val_loader, device="cpu", version=None, output_path=None, augs_idx=None, use_enhanced=False, trial=None):

    print("Starting SSL training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    lossfn = ContrastiveLoss(temperature).to(device)

    if use_enhanced:
        augmentfn = ControlledAugment(augs_idx, use_enhanced=use_enhanced)    #img_transforms_path="../../../data/upftfg26/apujols/processed")
    else:
        augmentfn = ControlledAugment(augs_idx, use_enhanced=use_enhanced)

    avg_loss = float("inf")
    patience_count = 0
    stop_epoch = num_epochs
    eval_every = 1
    debug = True
    best_acc = -float("inf")
    best_state = None

    history = pd.DataFrame(columns=["epoch", "schedule", "contrastive_loss", "val_accuracy", "train_accuracy", "uniformity", "alignment", "std", "time"])

    for epoch in range(num_epochs):

        training_start = time.time()
        model.train()
        total_loss = 0.0

        # Metrics        
        metrics = StreamingMetrics(alpha=2, t=2)

        for batch_idx, (images, fnames, bmins, bmaxs) in enumerate(ssl_train_loader):
            x_i, x_j = augmentfn(images, fnames, bmins, bmaxs)       # CPU augment

            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            # For debugging the image and its augmentations
            if debug and epoch == 0 and batch_idx == 0:
                os.makedirs(f"{output_path}", exist_ok=True)
                if epoch==0 and batch_idx==0:
                    imgs_orig = images.cpu().numpy()
                    imgs_i = x_i.cpu().numpy()
                    imgs_j = x_j.cpu().numpy()
                    print("Going to print the triplets...")
                    save_plot_augmentations(img_orig=imgs_orig[0, 0], img_i=imgs_i[0, 0], img_j=imgs_j[0, 0], save_path=output_path, version=version)
    
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = lossfn(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update alignment/uniformity
            metrics.update(z_i, z_j)

        scheduler.step()
        
        ssl_train_loss = total_loss / len(ssl_train_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - training_start
        
        # -----------------------------------
        # SSL Validation loss (optional)
        # -----------------------------------
        ssl_val_loss = None
        if ssl_val_loader is not None:
            ssl_val_loss = compute_val_loss(device, model, ssl_val_loader, augmentfn, lossfn)
        
        # -----------------------------------
        # Obtain epoch metrics
        # -----------------------------------
        epoch_alignment, epoch_uniformity = metrics.compute()

        # Train linear probe to get new accuracy
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            with torch.no_grad(): 
                train_feats, train_labels = extract_backbone_features(model, lp_train_loader, device)
                embeddings_std = float(train_feats.std(axis=0).mean())
                val_feats, val_labels = extract_backbone_features(model, lp_val_loader, device)

                lp_val_accuracy, lp_train_accuracy = run_linear_probe(train_feats, train_labels, val_feats, val_labels)
            
        
        contrastive_loss = ssl_val_loss if ssl_val_loss is not None else ssl_train_loss
        # Append to DataFrame
        history.loc[len(history)] = {
            "epoch": epoch + 1,
            "schedule": current_lr,
            "contrastive_loss": contrastive_loss,
            "val_accuracy": lp_val_accuracy,
            "train_accuracy": lp_train_accuracy,
            "uniformity": epoch_uniformity,
            "alignment": epoch_alignment,
            "std": embeddings_std,
            "time": epoch_time 
        }

        # ------------------------------
        # Early stopping
        # ------------------------------
        if ssl_val_loss is not None:
            improvement = (avg_loss - ssl_val_loss) / max(avg_loss, 1e-8)
            avg_loss = ssl_val_loss
        else:
            improvement = (avg_loss - ssl_train_loss) / max(avg_loss, 1e-8)
            avg_loss = ssl_train_loss
            
        if improvement < cutoff_ratio:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                stop_epoch = epoch + 1
                break
        else:
            patience_count = 0

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {contrastive_loss:.4f} | "
            f"LR: {current_lr:.6e} | "
            f"Δ: {improvement:.6f} | "
            f"Accuracy: {lp_val_accuracy:.6f} | "
            f"uniformity: {epoch_uniformity:.6f} | "
            f"alignment: {epoch_alignment:.6f} | "
            f"std: {embeddings_std:.6f}"
        )

        if lp_val_accuracy > best_acc:
            best_acc = lp_val_accuracy
            best_state = copy.deepcopy(model.state_dict())


        if trial is not None:
            trial.report(lp_val_accuracy, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    print("SSL training complete.\n")

    return model, best_acc, best_state, history, stop_epoch