import numpy as np
import pandas as pd
import copy
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.modules import MLPClassifier
from utils.checkpoint import save_checkpoint

def train_logreg(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, solver="liblinear"))
    ])

    param_grid = {
        "logreg__C": [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def train_linear_svm(X_train, y_train):
    pipe = Pipeline(
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(max_iter=5000))
    )

    param_grid = {
        "svm__C": [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        pipe, 
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def evaluate_linear_models(X_train, y_train):
    print("\n=== Logistic Regression ===")
    lr_model, lr_C, lr_acc = train_logreg(X_train, y_train)

    print("\n=== Linear SVM ===")
    svm_model, svm_C, svm_acc = train_linear_svm(X_train, y_train)

    print("\n=== Summary ===")
    print(f"LR best:  C={lr_C},  acc={lr_acc:.4f}")
    print(f"SVM best: C={svm_C}, acc={svm_acc:.4f}")

    return {
        "lr_model": lr_model,
        "svm_model": svm_model,
        "lr_acc": lr_acc,
        "svm_acc": svm_acc
    }

def get_split(cfg):
    X = np.load(cfg['paths']['features'])
    fnames = np.load(cfg['paths']['filenames'])

    fnames = fnames.astype(str)

    df = pd.read_csv(cfg['paths']['dataset'], sep=";")

    df = df[df['filename'].isin(fnames)]

    label_map = dict(zip(df['filename'], df['class']))

    y = np.array([label_map[f] for f in fnames])
    label_encoding = {"non-meteor": 0, "meteor": 1}
    y = np.array([label_encoding[label] for label in y])

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
    )

def train_linear_models(cfg):  
    X_train, _, y_train, _ = get_split(cfg)
    results = evaluate_linear_models(X_train=X_train, y_train=y_train)
    print(f"================ Linear Models ==================")
    print(f"lr_acc={results['lr_acc']}\tsvm_acc={results['svm_acc']}")

    output_dir = cfg['paths']['output_dir']
    VERSION = cfg['experiment_version']
    ckpt_path = os.path.join(output_dir, f"lr_model_{VERSION}.pt")
    save_checkpoint(results['lr_model'], ckpt_path)
    print(f"Saved LR model at {ckpt_path}")
    ckpt_path = os.path.join(output_dir, f"svm_model_{VERSION}.pt")
    save_checkpoint(results['svm_model'], ckpt_path)
    print(f"Saved SVM model at {ckpt_path}")

def train_mlp(model, train_loader, val_loader, device="cuda", epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0
    best_state = None
    patience = 5
    patience_counter = 0
    epoch_loss = 0

    history = pd.DataFrame(columns=["epoch", "loss", "val_accuracy", "train_accuracy"])

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total

        # Train
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        train_acc = correct / total


        print(f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Loss: {epoch_loss:.4f}")

        history.loc[len(history)] = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "val_accuracy": val_acc,
            "train_accuracy": train_acc
        }

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_acc


def train_mlp_classifier(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = get_split(cfg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)


    train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64,
    shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=64,
        shuffle=False
    )

    model = MLPClassifier(input_dim=X_train.shape[1])

    model, history, best_acc = train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        device=device
    )

    print(f"================ MLP ==================")
    print(f"mlp_acc={best_acc}")

    output_dir = cfg['paths']['output_dir']
    VERSION = cfg['experiment_version']
    ckpt_path = os.path.join(output_dir, f"mlp_model_{VERSION}.pt")
    save_checkpoint(model, ckpt_path)
    print(f"Saved MLP model at {ckpt_path}")
    history_path = os.path.join(output_dir, f"mlp_history_{VERSION}.csv")
    history.to_csv(history_path, sep=";", index=False)
    print(f"Saved MLP history at {history_path}")


def train_classifiers(cfg):

    print("STARTING CLASSIFIERS TRAINING\n")
    train_linear_models(cfg)
    print("\n")
    train_mlp_classifier(cfg)
    print("\nDONE")

