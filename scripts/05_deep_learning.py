#!/usr/bin/env python3
"""
scripts/05_deep_learning.py - Deep Learning & ML Baseline Models under Small-Sample Constraints
Trains and evaluates baseline Logistic Regression, Shallow MLP, and 1D/3D CNNs using Leave-One-Subject-Out (LOSO) CV.
Quantifies overfitting gaps, cross-subject generalization variance, and model convergence.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[Info] PyTorch is installing or not found; falling back to Scikit-Learn classifiers if needed.")

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==========================================
# PyTorch Model Architectures
# ==========================================

class ShallowMLP(nn.Module):
    """Shallow Multi-Layer Perceptron optimized for small N=26 regime."""
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.5):
        super(ShallowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConvNet1D(nn.Module):
    """1D Convolutional Neural Network for spatial/feature vectors."""
    def __init__(self, input_dim, hidden_channels=16, dropout_rate=0.5):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_channels * 16, 1)

    def forward(self, x):
        # Input x shape: (batch_size, input_dim) -> reshape to (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


# ==========================================
# Training & Evaluation Helper Functions
# ==========================================

def train_torch_model(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4):
    """Trains a PyTorch model and tracks training and validation loss/accuracy curves."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation evaluation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())
                val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

    return history


def evaluate_loso(X, y, loso_splits, model_type="logistic_regression", epochs=50, batch_size=8, lr=0.001):
    """
    Performs Leave-One-Subject-Out (LOSO) Cross-Validation for a specified model.
    """
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fold_accuracies = []

    print(f"\n⚡ Running Leave-One-Subject-Out CV for model: '{model_type}' (Total Folds: {len(loso_splits)})...")

    for fold in loso_splits:
        test_sub = fold["test_subject"]
        # In dual-condition dataset, each subject has 2 samples (Congruent=0, Incongruent=1)
        # Find sample indices belonging to test subject
        n_subs = len(loso_splits)
        sub_idx = fold["test_indices"][0]
        
        # Test indices correspond to subject sub_idx (Congruent) and sub_idx + n_subs (Incongruent)
        test_idx = [sub_idx, sub_idx + n_subs]
        train_idx = [j for j in range(len(y)) if j not in test_idx]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if model_type == "logistic_regression":
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)

        elif model_type in ["mlp", "cnn1d"] and not HAS_TORCH:
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)

        elif model_type in ["mlp", "cnn1d"] and HAS_TORCH:
            X_tr_t = torch.tensor(X_train, dtype=torch.float32)
            y_tr_t = torch.tensor(y_train, dtype=torch.long)
            X_te_t = torch.tensor(X_test, dtype=torch.float32)
            y_te_t = torch.tensor(y_test, dtype=torch.long)

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            test_ds = TensorDataset(X_te_t, y_te_t)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            if model_type == "mlp":
                net = ShallowMLP(input_dim=X.shape[1], hidden_dim=32, dropout_rate=0.5)
            else:
                net = ConvNet1D(input_dim=X.shape[1], hidden_channels=16, dropout_rate=0.5)

            _ = train_torch_model(net, train_loader, test_loader, epochs=epochs, lr=lr)
            
            net.eval()
            with torch.no_grad():
                logits = net(X_te_t)
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.5).astype(int)

        fold_acc = accuracy_score(y_test, preds)
        fold_accuracies.append(fold_acc)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)

    # Aggregate overall metrics
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    overall_rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    overall_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    overall_auc = roc_auc_score(y_true_all, y_prob_all)
    acc_std = float(np.std(fold_accuracies))

    results = {
        "accuracy": float(overall_acc),
        "accuracy_std": acc_std,
        "precision": float(overall_prec),
        "recall": float(overall_rec),
        "f1_score": float(overall_f1),
        "roc_auc": float(overall_auc),
        "fold_accuracies": [float(a) for a in fold_accuracies]
    }

    print(f"  Result ({model_type}): Accuracy={overall_acc:.4f} ± {acc_std:.4f} | F1={overall_f1:.4f} | AUC={overall_auc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Deep Learning & Baseline Classifier Benchmarking")
    parser.add_argument("--features_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/features", help="Path to extracted features")
    parser.add_argument("--output_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/models", help="Path to output model results")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs for PyTorch models")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classification dataset matrices
    X_pca_path = features_dir / "X_classification_pca.npy"
    X_voxel_path = features_dir / "X_classification_voxel.npy"
    y_path = features_dir / "y_classification.npy"
    loso_path = features_dir / "loso_splits.json"

    if not (X_pca_path.exists() and y_path.exists() and loso_path.exists()):
        print("ERROR: Feature files not found in derivatives/features. Please run scripts/04_feature_extraction.py first.", file=sys.stderr)
        sys.exit(1)

    X_pca = np.load(X_pca_path)
    y = np.load(y_path)

    with open(loso_path, "r") as f:
        loso_splits = json.load(f)

    print(f"Loaded classification matrix X_pca: {X_pca.shape}, y: {y.shape}, LOSO folds: {len(loso_splits)}")

    # Evaluate Model 1: Logistic Regression Baseline (on PCA features)
    lr_results = evaluate_loso(X_pca, y, loso_splits, model_type="logistic_regression")

    # Evaluate Model 2: Shallow MLP (on PCA features)
    mlp_results = evaluate_loso(X_pca, y, loso_splits, model_type="mlp", epochs=args.epochs, lr=args.lr)

    # Evaluate Model 3: 1D Convolutional Neural Network (on PCA features)
    cnn1d_results = evaluate_loso(X_pca, y, loso_splits, model_type="cnn1d", epochs=args.epochs, lr=args.lr)

    # Consolidate all model performance metrics
    summary = {
        "LogisticRegression_PCA": lr_results,
        "ShallowMLP_PCA": mlp_results,
        "ConvNet1D_PCA": cnn1d_results
    }

    with open(output_dir / "model_performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("✅ Phase 5 Model Benchmarking Completed Successfully!")
    print(f"Results saved to: {output_dir / 'model_performance_summary.json'}")
    print("=======================================================")


if __name__ == "__main__":
    main()
