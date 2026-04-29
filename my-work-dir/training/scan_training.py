import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformations.augment import ControlledAugmentGPU

def train_scan(model, params, args):
    num_epochs = params['num_epochs']
    loader = params['loader']
    val_loader = params['val_loader']          # <-- You will pass this from run_scan
    device = args['device']
    criterion = params['loss_fn']
    optimizer = params['optimizer']
    augmenter = params['augmenter']
    update_cluster_head_only = args['update_cluster_head_only']

    history = []   # <-- clean epoch-level history
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):

        # -------------------------
        # TRAINING (optimized)
        # -------------------------
        if update_cluster_head_only:
            model.eval()   # freeze BN
        else:
            model.train()

        train_total = 0.0
        train_consistency = 0.0
        train_entropy = 0.0
        num_batches = 0

        for batch in loader:
            anchors_raw = batch['anchor'].to(device, non_blocking=True)
            neighbors_raw = batch['neighbor'].to(device, non_blocking=True)

            bmins = batch['bmins'].to(device, non_blocking=True)
            bmaxs = batch['bmaxs'].to(device, non_blocking=True)

            # GPU augmentations (batched)
            anchors_aug = augmenter.one_view(anchors_raw, bmins, bmaxs)
            neighbors_aug = augmenter.one_view(neighbors_raw, bmins, bmaxs)

            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                if update_cluster_head_only:
                    with torch.no_grad():
                        anchors_features = model(anchors_aug, forward_pass='backbone')
                        neighbors_features = model(neighbors_aug, forward_pass='backbone')

                    anchors_output = model(anchors_features, forward_pass='head')
                    neighbors_output = model(neighbors_features, forward_pass='head')

                else:
                    anchors_output = model(anchors_aug)
                    neighbors_output = model(neighbors_aug)

                # anchors_output: list of H tensors [B, C]
                anchors_stack = torch.stack(anchors_output, dim=0)   # [H, B, C]
                neighbors_stack = torch.stack(neighbors_output, dim=0)

                # Flatten heads and batch → [H*B, C]
                a_flat = anchors_stack.reshape(-1, anchors_stack.size(-1))
                n_flat = neighbors_stack.reshape(-1, neighbors_stack.size(-1))

                # Single SCAN loss call for all heads
                total_loss, consistency_loss, entropy_loss = criterion(a_flat, n_flat)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate epoch metrics
            train_total += total_loss.item()
            train_consistency += consistency_loss.item()
            train_entropy += entropy_loss.item()
            num_batches += 1

        train_total /= num_batches
        train_consistency /= num_batches
        train_entropy /= num_batches


        # -------------------------
        # VALIDATION (optimized)
        # -------------------------
        model.eval()
        val_total = 0.0
        val_consistency = 0.0
        val_entropy = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                anchors_raw = batch['anchor'].to(device, non_blocking=True)
                neighbors_raw = batch['neighbor'].to(device, non_blocking=True)

                # Backbone forward (shared for all heads)
                anchors_features = model(anchors_raw, forward_pass='backbone')
                neighbors_features = model(neighbors_raw, forward_pass='backbone')

                # Head forward → list of H tensors [B, C]
                anchors_output = model(anchors_features, forward_pass='head')
                neighbors_output = model(neighbors_features, forward_pass='head')

                # Stack heads → shape [H, B, C]
                anchors_stack = torch.stack(anchors_output, dim=0)
                neighbors_stack = torch.stack(neighbors_output, dim=0)

                # Flatten heads and batch → shape [H*B, C]
                a_flat = anchors_stack.reshape(-1, anchors_stack.size(-1))
                n_flat = neighbors_stack.reshape(-1, neighbors_stack.size(-1))

                # Single SCAN loss call for all heads at once
                total_loss, consistency_loss, entropy_loss = criterion(a_flat, n_flat)

                val_total += total_loss.item()
                val_consistency += consistency_loss.item()
                val_entropy += entropy_loss.item()
                val_batches += 1

        val_total /= val_batches
        val_consistency /= val_batches
        val_entropy /= val_batches

        # Early stopping check
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0

            # Save best model
            best_state = {
                "backbone": model.backbone.state_dict(),
                "cluster_head": model.cluster_head.state_dict()
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


        # -------------------------
        # STORE EPOCH RECORD
        # -------------------------
        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_total,
            "train_consistency_loss": train_consistency,
            "train_entropy_loss": train_entropy,
            "val_total_loss": val_total,
            "val_consistency_loss": val_consistency,
            "val_entropy_loss": val_entropy
        })

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {train_total:.4f} | Val Loss: {val_total:.4f}")
        
    model.backbone.load_state_dict(best_state["backbone"])
    model.cluster_head.load_state_dict(best_state["cluster_head"])

    return model, pd.DataFrame(history)


def train_selflabel(model, train_loader, val_loader, optimizer, device, epoch, threshold=0.9):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    epoch_train_loss = 0.0
    epoch_train_sel = 0.0
    num_train_batches = 0

    augmenter = ControlledAugmentGPU(use_enhanced=True).to(device)

    for images, images_aug, labels, bmins, bmaxs in train_loader:
        images = images.to(device, non_blocking=True)
        bmins = bmins.to(device, non_blocking=True)
        bmaxs = bmaxs.to(device, non_blocking=True)

        # Strong augmentation (AMP)
        with torch.amp.autocast(device_type='cuda'):
            images_aug = augmenter.one_view(images, bmins, bmaxs)

        # 1. Pseudo-labels (no grad)
        with torch.amp.autocast(device_type='cuda'):
            logits = model(images, forward_pass='default')[0]
            probs = F.softmax(logits, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            mask = max_probs > threshold
            num_selected = mask.sum().item()

        # 2. Train only on confident samples
        batch_loss = 0.0
        if num_selected > 0:
            with torch.amp.autocast(device_type='cuda'):
                logits_aug = model(images_aug, forward_pass='default')[0]
                loss = criterion(logits_aug[mask], pseudo_labels[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = float(loss)

        epoch_train_loss += batch_loss
        epoch_train_sel += num_selected / images.size(0)
        num_train_batches += 1

    epoch_train_loss /= num_train_batches
    epoch_train_sel /= num_train_batches

    # -------------------------
    # VALIDATION (optimized)
    # -------------------------
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_sel = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for images, images_aug, labels, _, _ in val_loader:
            images = images.to(device, non_blocking=True)
            images_aug = images_aug.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                logits = model(images, forward_pass='default')[0]
                probs = F.softmax(logits, dim=1)
                max_probs, pseudo_labels = probs.max(dim=1)
                mask = max_probs > threshold
                num_selected = mask.sum().item()

                batch_loss = 0.0
                if num_selected > 0:
                    logits_aug = model(images_aug, forward_pass='default')[0]
                    loss = criterion(logits_aug[mask], pseudo_labels[mask])
                    batch_loss = float(loss)

            epoch_val_loss += batch_loss
            epoch_val_sel += num_selected / images.size(0)
            num_val_batches += 1

    epoch_val_loss /= num_val_batches
    epoch_val_sel /= num_val_batches


    # -------------------------
    # RETURN CLEAN EPOCH RECORD
    # -------------------------
    return {
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_selection_rate": epoch_train_sel,
        "val_loss": epoch_val_loss,
        "val_selection_rate": epoch_val_sel
    }
