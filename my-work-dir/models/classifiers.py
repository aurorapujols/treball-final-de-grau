import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

def train_logreg(X_train, y_train):
    pipe = Pipeline([
        ("logreg", LogisticRegression(max_iter=5000, solver='liblinear'))
    ])

    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=6,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def train_linear_svm(X_train, y_train):
    pipe = Pipeline([
        ("svm", LinearSVC(max_iter=5000))
    ])

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

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # ONE output
        )

    def forward(self, x):
        return self.net(x)  # raw logit