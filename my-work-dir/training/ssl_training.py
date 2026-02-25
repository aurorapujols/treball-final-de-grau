import os
import time
import optuna
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from losses.contrastive_loss import ContrastiveLoss
from transformations.augment import ControlledAugment, RandomAffineMeanFill
from evaluation.linear_probe import run_linear_probe
from evaluation.metrics import StreamingMetrics
from utils.plotting import save_plot_augmentations

def extract_backbone_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            h, _ = model(imgs)
            feats.append(h.cpu())
            labels.extend(lbls)

    feats = torch.cat(feats, dim=0).numpy()
    labels = np.array(labels)
    return feats, labels

def train_ssl(model, batch_size, num_epochs, patience, cutoff_ratio, lr, temperature, loader, train_loader, val_loader, device="cpu", version=None, output_path=None, use_image_augs=False, trial=None):

    print("Starting SSL training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    lossfn = ContrastiveLoss(temperature).to(device)

    if use_image_augs:
        augmentfn = ControlledAugment(img_transforms_path="../../../data/upftfg26/apujols/processed")
    else:
        augmentfn = ControlledAugment()

    avg_loss = float("inf")
    patience_count = 0
    stop_epoch = num_epochs
    eval_every = 1
    debug = False

    history = pd.DataFrame(columns=["epoch", "schedule", "contrastive_loss", "accuracy", "uniformity", "alignment", "time"])

    for epoch in range(num_epochs):

        training_start = time.time()
        model.train()
        total_loss = 0.0

        # Metrics        
        metrics = StreamingMetrics(alpha=2, t=2)

        for batch_idx, (images, fnames) in enumerate(loader):
            x_i, x_j = augmentfn(images, fnames)       # CPU augment

            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            # For debugging the image and its augmentations
            if debug and epoch == 0 and batch_idx == 0:
                os.makedirs(f"{output_path}", exist_ok=True)
                if epoch==0 and batch_idx==0:
                    imgs_orig = images.cpu().numpy()
                    imgs_i = x_i.cpu().numpy()
                    imgs_j = x_j.cpu().numpy()
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
        epoch_loss = total_loss / len(loader)
        improvement = (avg_loss - epoch_loss) / max(avg_loss, 1e-8)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - training_start

        # -----------------------------------
        # Obtain epoch metrics
        # -----------------------------------

        epoch_alignment, epoch_uniformity = metrics.compute()

        linear_probe_acc = history["accuracy"].iloc[-1] if len(history) > 0 else 0.0

        # Train linear probe to get new accuracy
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            with torch.no_grad(): 
                train_feats, train_labels = extract_backbone_features(model, train_loader, device)
                val_feats, val_labels = extract_backbone_features(model, val_loader, device)
                linear_probe_acc, _ = run_linear_probe(train_feats, train_labels, val_feats, val_labels)
            

        # Append to DataFrame
        history.loc[len(history)] = {
            "epoch": epoch + 1,
            "schedule": current_lr,
            "contrastive_loss": epoch_loss,
            "accuracy": linear_probe_acc,
            "uniformity": epoch_uniformity,
            "alignment": epoch_alignment,
            "time": epoch_time 
        }

        print(f"Epoch {epoch:03d} | "f"Loss: {epoch_loss:.4f} | "f"LR: {current_lr:.6e} | "f"Δ: {improvement:.6f} | "f"Accuracy: {linear_probe_acc}")

        if improvement < cutoff_ratio:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                stop_epoch = epoch + 1
                break
        else:
            patience_count = 0

        avg_loss = epoch_loss

        if trial is not None:
            trial.report(linear_probe_acc, epoch)

            # if trial.should_prune():
            #     raise optuna.TrialPruned()

    print("SSL training complete.\n")

    return model, history, stop_epoch