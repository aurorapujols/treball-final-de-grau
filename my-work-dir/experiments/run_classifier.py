import os
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
 
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
 
import models.ssl_model as encoder
import models.classifiers as classifiers
import data.datasets as datasets
import transformations.transform as transform
from data.dataloaders import get_ssl_loader


def train_linear_models(X_train, X_val, y_train, y_val):  

    print("\nTraining Logistic Regression ...")
    lr_model, lr_C, lr_acc = classifiers.train_logreg(X_train, y_train)
    
    print("\nTraining Support Vector Machine ...")
    svm_model, svm_C, svm_acc = classifiers.train_linear_svm(X_train, y_train)
    
    print("\n=== Summary ===")
    print(f"LR best:  C={lr_C},  acc={lr_acc:.4f}")
    print(f"SVM best: C={svm_C}, acc={svm_acc:.4f}")
    y_pred_lr = lr_model.predict(X_val)
    y_pred_svm = svm_model.predict(X_val)
    print(f"(Validation Accuracy)    Logistic Regression: {np.sum(y_pred_lr == y_val)/len(y_pred_lr):.4f} | Support Vector Machine: {np.sum(y_pred_svm == y_val)/len(y_pred_svm):.4f}")

    return lr_model, svm_model

def train_mlp(X_train, X_val, y_train, y_val, epochs=50, batch_size=64, lr=5e-5, device="cuda"):
    
    print("\nTraining Multilayer Perceptron ...")
    model = classifiers.MLPClassifier(input_dim=X_train.shape[1])

    best_val_acc = 0.0
    best_model_state = None

    # Convert numpy → torch
    X_train = torch.tensor(X_train, dtype=torch.float32)    
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)
    
    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # with L2 regularization

    model.to(device)
    
    history = pd.DataFrame(columns=["epoch", "val_loss", "train_loss", "val_accuracy", "train_accuracy"])

    # Train loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()


        # Store in history
        history.loc[len(history)] = [
            epoch, val_loss, train_loss, val_acc, train_acc
        ]

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    print(f"================ MLP ==================")
    print(f"mlp_acc={history['val_accuracy'].max()}")

    model.load_state_dict(best_model_state)

    return model, history


# ------------------------------------
# NEW: Cross-validation
# ------------------------------------

def _train_mlp_fold(X_train, y_train, X_val, y_val, input_dim,
                    epochs=60, batch_size=64, lr=5e-5, device="cuda"):
    """Train one MLP fold and return best validation accuracy."""
    model = classifiers.MLPClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss()(model(xb), yb).backward()
            optimizer.step()

    # Evaluate on fold val set
    model.eval()
    with torch.no_grad():
        logits = model(X_v.to(device))
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        acc = (preds.squeeze() == torch.tensor(y_val, dtype=torch.float32)).float().mean().item()

    return acc


def cross_validate_classifiers(X, y, n_splits=5, mlp_epochs=60,
                                mlp_batch_size=64, mlp_lr=5e-5, device="cuda"):
    """
    Run stratified k-fold cross-validation on all three classifiers
    using the full (train+val) embeddings X and labels y.

    Returns a DataFrame with per-fold and summary (mean ± std) results,
    and prints a paired t-test comparing LR vs MLP.
    """
    print(f"\n{'='*50}")
    print(f"Cross-Validation  ({n_splits}-fold Stratified)")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr_scores, svm_scores, mlp_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        # --- Logistic Regression ---
        lr_pipe = Pipeline([("lr", LogisticRegression(C=10.0, max_iter=5000, solver='liblinear'))])
        lr_pipe.fit(X_tr, y_tr)
        lr_acc = lr_pipe.score(X_v, y_v)
        lr_scores.append(lr_acc)

        # --- Linear SVM ---
        svm_pipe = Pipeline([("svm", LinearSVC(C=1.0, max_iter=5000))])
        svm_pipe.fit(X_tr, y_tr)
        svm_acc = svm_pipe.score(X_v, y_v)
        svm_scores.append(svm_acc)

        # --- MLP ---
        mlp_acc = _train_mlp_fold(X_tr, y_tr, X_v, y_v,
                                   input_dim=X.shape[1],
                                   epochs=mlp_epochs,
                                   batch_size=mlp_batch_size,
                                   lr=mlp_lr,
                                   device=device)
        mlp_scores.append(mlp_acc)

        print(f"  LR:  {lr_acc:.4f} | SVM: {svm_acc:.4f} | MLP: {mlp_acc:.4f}")

    lr_scores  = np.array(lr_scores)
    svm_scores = np.array(svm_scores)
    mlp_scores = np.array(mlp_scores)

    # --- Summary table ---
    results_df = pd.DataFrame({
        "fold":        list(range(1, n_splits + 1)) + ["mean", "std"],
        "LR":          list(lr_scores)  + [lr_scores.mean(),  lr_scores.std()],
        "SVM":         list(svm_scores) + [svm_scores.mean(), svm_scores.std()],
        "MLP":         list(mlp_scores) + [mlp_scores.mean(), mlp_scores.std()],
    })

    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print(f"{'='*50}")
    print(f"  LR  : {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  SVM : {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
    print(f"  MLP : {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}")

    # --- Paired t-test: LR vs MLP ---
    t_stat, p_value = stats.ttest_rel(lr_scores, mlp_scores)
    print(f"\nPaired t-test (LR vs MLP): t={t_stat:.4f}, p={p_value:.4f}")
    if p_value > 0.05:
        print("  → No statistically significant difference (p > 0.05). Prefer the simpler linear model.")
    else:
        print("  → Statistically significant difference (p ≤ 0.05).")

    return results_df


def train_classifiers(cfg):
    print("STARTING CLASSIFIERS TRAINING\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = cfg['paths']['output_dir']
    batch_size = cfg['batch_size']
    VERSION = cfg['experiment_version']


    # ----------------------------------
    # Load model and loaders
    # ----------------------------------
    ssl_model = encoder.get_model(cfg['ssl_model_path'])   # loads the model with final architecture

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

    print(f"Dataset: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")


    # ----------------------------------
    # Extract features
    # ----------------------------------
    X_train, _, y_train = encoder.get_encoding_and_projection(
        model=ssl_model,
        dataloader=train_loader,
        device=device
    )
    X_val, _, y_val = encoder.get_encoding_and_projection(
        model=ssl_model,
        dataloader=val_loader,
        device=device
    )

    label_map = {"unknown": 0, "meteor": 1}
    y_train = np.array([label_map[y] for y in y_train], dtype=np.int64)
    y_val   = np.array([label_map[y] for y in y_val],   dtype=np.int64)


    # ------------------------------------
    # Cross-validation (train+val pooled)
    # ------------------------------------
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    cv_results = cross_validate_classifiers(
        X_all, y_all,
        n_splits=5,
        mlp_epochs=60,
        mlp_batch_size=64,
        mlp_lr=5e-5,
        device=device
    )
    cv_results.to_csv(
        os.path.join(output_path, f"cv_results_{VERSION}.csv"), sep=";", index=False
    )


    # ------------------------------------
    # Train final models on full train set
    # ------------------------------------
    lr_model, svm_model = train_linear_models(X_train, X_val, y_train, y_val)
    mlp_model, history  = train_mlp(X_train, X_val, y_train, y_val, epochs=60)


    # ------------------------------------
    # Save results and models
    # ------------------------------------
    history.to_csv(os.path.join(output_path, f"history_mlp_model{VERSION}.csv"), sep=";")

    joblib.dump(lr_model,  os.path.join(output_path, f"lr_model_{VERSION}.pt"))
    joblib.dump(svm_model, os.path.join(output_path, f"svm_model_{VERSION}.pt"))
    joblib.dump(mlp_model, os.path.join(output_path, f"mlp_model_{VERSION}.pt"))

    print("\nDONE")