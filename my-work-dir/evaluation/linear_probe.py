import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_linear_probe(train_feats, train_labels, val_feats, val_labels, max_iters=5000):
    """
    Fits a logistic regression classifier on frozen backbone features
    and evaluates it on validation features.
    """
    clf = LogisticRegression(max_iter=max_iters)
    clf.fit(train_feats, train_labels)

    val_pred = clf.predict(val_feats)
    train_pred = clf.predict(train_feats)
    val_acc = accuracy_score(val_labels, val_pred)
    train_acc = accuracy_score(train_labels, train_pred)

    return val_acc, train_acc
