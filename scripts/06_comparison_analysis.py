#!/usr/bin/env python3
"""
scripts/06_comparison_analysis.py - Phase 6: Comparative & Spatial Attribution Evaluation
Generates quantitative model comparisons, spatial attribution maps (Weights / Saliency),
computes Dice coefficient & Pearson correlation with GLM Z-maps, and performs permutation testing.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def dice_coefficient(map1, map2, threshold_percentile=90):
    """Calculates Dice overlap coefficient between two thresholded spatial maps."""
    thresh1 = np.percentile(np.abs(map1), threshold_percentile)
    thresh2 = np.percentile(np.abs(map2), threshold_percentile)
    
    bin1 = (np.abs(map1) >= thresh1).astype(int)
    bin2 = (np.abs(map2) >= thresh2).astype(int)
    
    intersection = np.sum(bin1 * bin2)
    total = np.sum(bin1) + np.sum(bin2)
    
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_permutation_test(X_raw, y, loso_splits, n_permutations=1000, n_components=20, seed=42):
    """
    Performs non-parametric permutation testing by shuffling labels N times
    to construct an empirical null distribution using fold-nested scaling & PCA.
    """
    np.random.seed(seed)
    null_accuracies = []
    print(f"\n🎲 Running {n_permutations} permutations for non-parametric significance testing...")

    for p in range(n_permutations):
        if (p + 1) % 200 == 0 or p == 0:
            print(f"  [Permutation Progress] {p + 1} / {n_permutations} completed...")
            
        y_shuffled = np.random.permutation(y)
        y_true_all, y_pred_all = [], []

        for fold in loso_splits:
            n_subs = len(loso_splits)
            sub_idx = fold["test_indices"][0]
            test_idx = [sub_idx, sub_idx + n_subs]
            train_idx = [j for j in range(len(y)) if j not in test_idx]

            X_tr_raw, y_tr = X_raw[train_idx], y_shuffled[train_idx]
            X_te_raw, y_te = X_raw[test_idx], y_shuffled[test_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            X_te_scaled = scaler.transform(X_te_raw)

            n_comp = min(n_components, X_tr_scaled.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr = pca.fit_transform(X_tr_scaled)
            X_te = pca.transform(X_te_scaled)

            clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            y_true_all.extend(y_te)
            y_pred_all.extend(preds)

        null_acc = accuracy_score(y_true_all, y_pred_all)
        null_accuracies.append(null_acc)

    return np.array(null_accuracies)


from scipy.stats import ttest_rel


def compute_bootstrap_ci(values, n_bootstraps=1000, ci=95, seed=42):
    """Computes non-parametric bootstrap confidence interval for a metric vector."""
    np.random.seed(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_bootstraps):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2.0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return lower, upper


def compute_cohens_d(x1, x2):
    """Computes Cohen's d effect size for paired samples."""
    diff = x1 - x2
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Comparative Analysis & Spatial Attribution")
    parser.add_argument("--features_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/features", help="Path to features")
    parser.add_argument("--models_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/models", help="Path to models summary")
    parser.add_argument("--output_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/results", help="Path to output figures & results")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of label permutation shuffles")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 5 model summary
    summary_file = models_dir / "model_performance_summary.json"
    if not summary_file.exists():
        print(f"ERROR: Model summary not found at {summary_file}", file=sys.stderr)
        sys.exit(1)

    with open(summary_file, "r") as f:
        model_summary = json.load(f)

    # 1. Compute Statistical Model Comparisons (Paired t-tests, CIs, Effect Sizes)
    models = list(model_summary.keys())
    stats_comparison = {}
    
    for m in models:
        accs = np.array(model_summary[m]["fold_accuracies"])
        low, high = compute_bootstrap_ci(accs)
        model_summary[m]["accuracy_95ci"] = [float(low), float(high)]

    lr_accs = np.array(model_summary["LogisticRegression_PCA"]["fold_accuracies"])
    mlp_accs = np.array(model_summary["ShallowMLP_PCA"]["fold_accuracies"])
    cnn_accs = np.array(model_summary["ConvNet1D_PCA"]["fold_accuracies"])

    # Paired t-tests & Cohen's d
    t_lr_mlp, p_lr_mlp = ttest_rel(lr_accs, mlp_accs)
    d_lr_mlp = compute_cohens_d(lr_accs, mlp_accs)

    t_lr_cnn, p_lr_cnn = ttest_rel(lr_accs, cnn_accs)
    d_lr_cnn = compute_cohens_d(lr_accs, cnn_accs)

    stats_comparison = {
        "LR_vs_MLP": {"t_statistic": float(t_lr_mlp), "p_value": float(p_lr_mlp), "cohens_d": float(d_lr_mlp)},
        "LR_vs_CNN": {"t_statistic": float(t_lr_cnn), "p_value": float(p_lr_cnn), "cohens_d": float(d_lr_cnn)}
    }

    print("\n📊 Paired Model Comparison Tests:")
    print(f"  Logistic Regression vs. Shallow MLP: t = {t_lr_mlp:.3f}, p = {p_lr_mlp:.4f}, Cohen's d = {d_lr_mlp:.3f}")
    print(f"  Logistic Regression vs. 1D-CNN: t = {t_lr_cnn:.3f}, p = {p_lr_cnn:.4f}, Cohen's d = {d_lr_cnn:.3f}")

    # 2. Plot Model Comparison Bar Chart
    accs = [model_summary[m]["accuracy"] for m in models]
    stds = [model_summary[m]["accuracy_std"] for m in models]
    aucs = [model_summary[m]["roc_auc"] for m in models]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, accs, width, yerr=stds, label='LOSO Accuracy', capsize=5, color='#2b5c8f')
    plt.bar(x + width/2, aucs, width, label='ROC-AUC', color='#e07a5f')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance Level (0.50)')

    plt.ylabel('Score')
    plt.title('Model Generalization Performance under N=26 Small-Sample Regime')
    plt.xticks(x, [m.replace('_PCA', '') for m in models])
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance_comparison.png", dpi=300)
    plt.close()

    # 3. Fold Accuracy Distribution Box Plot with Individual Fold Jitter Points
    plt.figure(figsize=(9, 5))
    fold_data = [model_summary[m]["fold_accuracies"] for m in models]
    labels_clean = [m.replace('_PCA', '') for m in models]
    
    bp = plt.boxplot(fold_data, tick_labels=labels_clean, patch_artist=True,
                     boxprops=dict(facecolor='#2b5c8f', alpha=0.5),
                     medianprops=dict(color='#e07a5f', linewidth=2.5))
    
    # Overlay individual fold points with jitter
    np.random.seed(42)
    for i, folds in enumerate(fold_data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(folds))
        plt.scatter(np.full_like(folds, i) + jitter, folds, alpha=0.7, color='#2b5c8f', edgecolors='black', linewidths=0.5, s=35, label='Subject Fold' if i == 1 else "")

    plt.axhline(0.5, color='gray', linestyle='--', label='Chance Level (0.50)')
    plt.ylabel('Fold Accuracy')
    plt.title('Fold-Level Accuracy Distribution across 26 LOSO Subject Folds')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_accuracy_distribution.png", dpi=300)
    plt.close()

    # 4. Spatial Attribution vs. GLM Z-Map Overlap Analysis
    X_voxel = np.load(features_dir / "X_classification_voxel.npy")
    y = np.load(features_dir / "y_classification.npy")
    with open(features_dir / "loso_splits.json", "r") as f:
        loso_splits = json.load(f)

    # Reconstruct whole-brain spatial voxel weights via PCA back-projection: w_voxel = V_pca * w_pca
    scaler_full = StandardScaler()
    X_voxel_scaled = scaler_full.fit_transform(X_voxel)
    
    pca_full = PCA(n_components=20, random_state=42)
    X_pca_full = pca_full.fit_transform(X_voxel_scaled)
    
    clf_pca = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_pca.fit(X_pca_full, y)
    
    w_pca = clf_pca.coef_[0]
    spatial_weights = pca_full.components_.T @ w_pca

    # Load average GLM voxel Z-stat vector (Incongruent > Congruent)
    glm_zstat_voxels = np.load(features_dir / "features_voxel_zstat.npy").mean(axis=0)

    # Compute spatial overlap metrics
    corr_val, p_val = pearsonr(spatial_weights, glm_zstat_voxels)
    safe_p_val = float(p_val) if p_val > 0 else 1.0e-15
    dice_val = dice_coefficient(spatial_weights, glm_zstat_voxels, threshold_percentile=90)

    print("\n🗺️ Spatial Correspondence (GLM Z-Map vs. PCA Reconstructed ML Attribution):")
    print(f"  Pearson Spatial Correlation (r): {corr_val:.4f} (p = {safe_p_val:.4e})")
    print(f"  Dice Overlap Coefficient (Top 10% Voxels): {dice_val:.4f}")

    # 5. Non-Parametric Permutation Null Testing (Fold-Nested 1000 Permutations)
    null_accs = run_permutation_test(X_voxel, y, loso_splits, n_permutations=args.n_permutations)
    obs_acc = model_summary["LogisticRegression_PCA"]["accuracy"]
    p_empirical = (np.sum(null_accs >= obs_acc) + 1) / (len(null_accs) + 1)

    print(f"\n📊 Permutation Test Result (N={args.n_permutations}):")
    print(f"  Observed LOSO Accuracy: {obs_acc:.4f}")
    print(f"  Empirical Null Mean Accuracy: {null_accs.mean():.4f} ± {null_accs.std():.4f}")
    print(f"  Empirical p-value: {p_empirical:.4f}")

    # Plot Permutation Null Distribution
    plt.figure(figsize=(9, 5))
    plt.hist(null_accs, bins=25, color='#2b5c8f', alpha=0.7, edgecolor='black', label=f'Null Distribution (Mean = {null_accs.mean():.2%})')
    plt.axvline(obs_acc, color='#e07a5f', linestyle='--', linewidth=2.5, label=f'Observed Accuracy ({obs_acc:.2%}, p = {p_empirical:.4f})')
    plt.xlabel('LOSO Classification Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Non-Parametric Permutation Null Distribution ({args.n_permutations} Shuffles)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_null_distribution.png", dpi=300)
    plt.close()

    # Save final evaluation report
    final_report = {
        "model_performance": model_summary,
        "statistical_comparisons": stats_comparison,
        "spatial_correspondence": {
            "pearson_correlation": float(corr_val),
            "pearson_pvalue": float(safe_p_val),
            "dice_coefficient_top10pct": float(dice_val),
            "methodology": "Linear model coefficients fitted on standardized in-mask voxels correlated with unthresholded group GLM Z-stat map."
        },
        "permutation_test": {
            "n_permutations": args.n_permutations,
            "observed_accuracy": float(obs_acc),
            "null_accuracy_mean": float(null_accs.mean()),
            "null_accuracy_std": float(null_accs.std()),
            "empirical_p_value": float(p_empirical)
        }
    }

    with open(output_dir / "final_study_results.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n=======================================================")
    print("🎉 Phase 6 Comparative & Spatial Analysis Completed!")
    print(f"Final results report: {output_dir / 'final_study_results.json'}")
    print(f"Performance plots saved in: {output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
