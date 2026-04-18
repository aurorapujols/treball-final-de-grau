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

from losses.losses import ContrastiveLoss, SupervisedContrastiveLoss
from transformations.augment import ControlledAugment, ControlledAugmentGPU
from evaluation.linear_probe import run_linear_probe
from evaluation.metrics import StreamingMetrics
from utils.plotting import save_plot_augmentations

def extract_backbone_features(model, dataloader, device):
    model.eval()
    feats, labels, filenames = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, fnames, _, _, lbls = batch
            imgs = imgs.to(device)
            h, _ = model(imgs)
            feats.append(h.cpu())
            labels.extend(lbls)
            filenames.extend(fnames)

    feats = torch.cat(feats, dim=0).numpy()
    labels = np.array(labels)
    return feats, labels, filenames
    
def compute_loss(device, model, loader, augmentfn, lossfn):
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for images, fnames, bmins, bmaxs, labels in loader:
            x_i, x_j = augmentfn(images, fnames, bmins, bmaxs)
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            val_total += lossfn(z_i, z_j).item()
            
        ssl_val_loss = val_total / len(loader)
    
    return ssl_val_loss

def compute_loss_gpu(device, model, loader, augmentfn, lossfn):
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for images, fnames, bmins, bmaxs, labels_str in loader:
            images = images.to(device, non_blocking=True)
            bmins = bmins.to(device, non_blocking=True).float()
            bmaxs = bmaxs.to(device, non_blocking=True).float()

            x_i, x_j = augmentfn(images, fnames, bmins, bmaxs)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            if isinstance(lossfn, ContrastiveLoss):
                val_total += lossfn(z_i, z_j).item()
            if isinstance(lossfn, SupervisedContrastiveLoss):
                label_map = {"non-meteor": 0, "meteor": 1}
                labels = torch.tensor([label_map[l] for l in labels_str], device=device)   # (b,) tensor with 1s and 0s for meteor and non-meteor class, respectively
                val_total += lossfn(z_i, z_j, labels).item()

    ssl_val_loss = val_total / len(loader)
    return ssl_val_loss

def train_ssl(model, params, loaders, args, trial=None):

    print("Starting SSL training...")
    device = args['device']

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.7, classical momentum)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    loss = params['loss']
    lossfn = None
    if loss == "contrastive_loss":
        lossfn = ContrastiveLoss(temperature=params['temperature']).to(device)
    elif loss == "supervised_contrastive_loss":
        lossfn = SupervisedContrastiveLoss(temperature=params['temperature']).to(device)

    use_enhanced = args['use_enhanced']
    augs_idx = args['augs_idx']
    
    augmentfn = ControlledAugmentGPU(augs_idx=augs_idx, use_enhanced=use_enhanced).to(device)

    avg_loss = float("inf")
    patience_count = 0
    stop_epoch = params['num_epochs']
    eval_every = args['eval_every']
    best_acc = -float("inf")
    best_state = None
    debug = True

    history = pd.DataFrame(columns=["epoch", "schedule", "val_loss", "train_loss", "val_accuracy", "train_accuracy", "uniformity", "alignment", "std", "time"])

    for epoch in range(params['num_epochs']):

        training_start = time.time()
        model.train()
        total_loss = 0.0

        # Metrics        
        metrics = StreamingMetrics(alpha=2, t=2)

        for batch_idx, (images, fnames, bmins, bmaxs, labels_str) in enumerate(loaders['train_loader']):
            images = images.to(device, non_blocking=True)   # (B,1,128,128) in [0,1]
            bmins = bmins.to(device, non_blocking=True).float()
            bmaxs = bmaxs.to(device, non_blocking=True).float()
            
            x_i, x_j = augmentfn(images, fnames, bmins, bmaxs)       # GPU augment

            # x_i = x_i.to(device, non_blocking=True)
            # x_j = x_j.to(device, non_blocking=True)

            # For debugging the image and its augmentations
            if debug and epoch == 0 and batch_idx < 5:
                os.makedirs(f"{args['output_path']}/debug_augs", exist_ok=True)

                imgs_orig = images.cpu().numpy()
                imgs_i = x_i.cpu().numpy()
                imgs_j = x_j.cpu().numpy()

                N = min(2, images.size(0))  # print 2 samples per batch

                for k in range(N):
                    save_plot_augmentations(
                        img_orig=imgs_orig[k, 0],
                        img_i=imgs_i[k, 0],
                        img_j=imgs_j[k, 0],
                        save_path=f"{args['output_path']}batch{batch_idx}_sample{k}.png",
                        version=args['version']
                    )

            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            if isinstance(lossfn, ContrastiveLoss):
                loss = lossfn(z_i, z_j)
            elif isinstance(lossfn, SupervisedContrastiveLoss):
                label_map = {"non-meteor": 0, "meteor": 1}
                labels = torch.tensor([label_map[l] for l in labels_str], device=device)   # (b,) tensor with 1s and 0s for meteor and non-meteor class, respectively
                loss = lossfn(z_i, z_j, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update alignment/uniformity
            metrics.update(z_i, z_j)

        scheduler.step()
        
        ssl_train_loss = total_loss / len(loaders['train_loader'])
        
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - training_start
        
        # -----------------------------------
        # SSL Validation loss (optional)
        # -----------------------------------
        ssl_val_loss = None
        if loaders['val_loader'] is not None:
            ssl_val_loss = compute_loss_gpu(device, model, loaders['val_loader'], augmentfn, lossfn)
        
        # -----------------------------------
        # Obtain epoch metrics
        # -----------------------------------
        epoch_alignment, epoch_uniformity = metrics.compute()

        # Train linear probe to get new accuracy
        lp_val_accuracy = float("nan")
        lp_train_accuracy = float("nan")
        embeddings_std = float("nan")
        if epoch % eval_every == 0 or epoch == params['num_epochs'] - 1:
            with torch.no_grad(): 
                train_feats, train_labels, _ = extract_backbone_features(model, loaders['train_loader'], device)
                embeddings_std = float(train_feats.std(axis=0).mean())
                val_feats, val_labels, _ = extract_backbone_features(model, loaders['val_loader'], device)

                clf, lp_val_accuracy, lp_train_accuracy = run_linear_probe(train_feats, train_labels, val_feats, val_labels)
        
        # Append to DataFrame
        history.loc[len(history)] = {
            "epoch": epoch + 1,
            "schedule": current_lr,
            "val_loss": ssl_val_loss,
            "train_loss": ssl_train_loss,
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
        # generalization gap & improvement
        gap = (ssl_train_loss - ssl_val_loss) / max(ssl_val_loss, 1e-8)
        improvement = (avg_loss - ssl_val_loss) / max(avg_loss, 1e-8)
        
        if gap > params["max_gap"] or improvement < params["cutoff_ratio"]:
            patience_count += 1
            if patience_count >= params['patience']:
                print(f"Early stopping triggered. Gap={gap:.4f}")
                stop_epoch = epoch + 1
                break
        else:
            patience_count = 0
        avg_loss = ssl_val_loss

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {ssl_train_loss:.4f} | "
            f"ValLoss: {ssl_val_loss:.4f} | "
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