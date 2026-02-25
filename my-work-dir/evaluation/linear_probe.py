import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_linear_probe(train_feats, train_labels, val_feats, val_labels):
    """
    Fits a logistic regression classifier on frozen backbone features
    and evaluates it on validation features.
    """
    clf = LogisticRegression(max_iter=5000)
    clf.fit(train_feats, train_labels)

    val_pred = clf.predict(val_feats)
    acc = accuracy_score(val_labels, val_pred)

    return acc, clf
