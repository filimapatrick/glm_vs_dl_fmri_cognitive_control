#!/usr/bin/env python3
"""
scripts/04_feature_extraction.py - Feature Extraction for Machine Learning & Deep Learning
Extracts whole-brain voxel contrast maps, ROI activation summaries, PCA dimensionality-reduced embeddings,
and generates Leave-One-Subject-Out (LOSO) cross-validation fold partitions for N=26 regime.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford

# Locate FSLDIR from environment or default
FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:
    default_fsl = Path("/Users/patrickfilima/fsl")
    if default_fsl.exists():
        os.environ["FSLDIR"] = str(default_fsl)
        os.environ["PATH"] = f"{default_fsl}/bin:" + os.environ["PATH"]
        FSLDIR = str(default_fsl)
    else:
        print("ERROR: FSLDIR environment variable not found. Please set FSLDIR.", file=sys.stderr)
        sys.exit(1)

# Standard space brain mask for MNI152 2mm
MNI_MASK = Path(FSLDIR) / "data" / "standard" / "MNI152_T1_2mm_brain_mask.nii.gz"
if not MNI_MASK.exists():
    print(f"ERROR: MNI152 brain mask not found at {MNI_MASK}", file=sys.stderr)
    sys.exit(1)


def load_subject_contrast_maps(qc_subjects, glm_dir, contrast_name="IncongruentGreaterThanCongruent", map_type="zstat"):
    """
    Loads contrast NIfTI files across all QC-passed subjects.
    
    Parameters:
        qc_subjects (list): List of subject IDs (e.g. ['sub-01', 'sub-02', ...])
        glm_dir (Path): Path to derivatives/glm
        contrast_name (str): Contrast identifier string
        map_type (str): 'zstat' or 'cope'
        
    Returns:
        file_paths (list of Path): List of existing contrast map file paths
        available_subjects (list of str): List of matched subject IDs
    """
    file_paths = []
    available_subjects = []
    
    for sub in qc_subjects:
        file_path = glm_dir / sub / f"{sub}_contrast-{contrast_name}_{map_type}.nii.gz"
        if file_path.exists():
            file_paths.append(str(file_path))
            available_subjects.append(sub)
        else:
            print(f"  [Warning] Missing {map_type} map for {sub} at {file_path}")
            
    return file_paths, available_subjects


def extract_voxel_features(file_paths, mask_path):
    """
    Extracts whole-brain masked 1D voxel vectors for each subject map.
    
    Returns:
        voxel_matrix (np.ndarray): Shape (N_subjects, N_voxels)
        masker (NiftiMasker): Fitted nilearn masker object
    """
    print(f"\n🧠 Extracting whole-brain voxel features using mask: {mask_path.name}")
    masker = NiftiMasker(mask_img=str(mask_path), standardize=False, memory='nilearn_cache', verbose=0)
    masker.fit()
    
    voxel_matrix = masker.transform(file_paths)
    print(f"  Extracted voxel matrix shape: {voxel_matrix.shape} (Subjects x Voxels)")
    return voxel_matrix, masker


def extract_pca_features(voxel_matrix, n_components=20):
    """
    Applies Principal Component Analysis (PCA) to reduce whole-brain voxel dimensions.
    
    Returns:
        pca_matrix (np.ndarray): Shape (N_subjects, n_components)
        pca (PCA): Fitted PCA object
    """
    n_samples = voxel_matrix.shape[0]
    n_comp = min(n_components, n_samples - 1)
    print(f"\n📊 Performing PCA dimensionality reduction (Components: {n_comp})...")
    
    pca = PCA(n_components=n_comp, random_state=42)
    pca_matrix = pca.fit_transform(voxel_matrix)
    
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA matrix shape: {pca_matrix.shape}")
    print(f"  Total explained variance ratio: {explained_var:.2f}%")
    return pca_matrix, pca


def extract_roi_features(file_paths, available_subjects):
    """
    Extracts mean activation z-scores/copes across anatomical ROIs using the Harvard-Oxford atlas.
    
    Returns:
        roi_df (pd.DataFrame): DataFrame of shape (N_subjects, N_ROIs)
    """
    print("\n🏛️ Extracting ROI-based features using Harvard-Oxford Cortical Atlas...")
    try:
        ho_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = ho_atlas.maps
        labels = ho_atlas.labels[1:]  # Exclude background label
        
        labels_masker = NiftiLabelsMasker(labels_img=atlas_img, labels=labels, standardize=False)
        roi_matrix = labels_masker.fit_transform(file_paths)
        
        roi_df = pd.DataFrame(roi_matrix, index=available_subjects, columns=labels)
        print(f"  Extracted ROI matrix shape: {roi_matrix.shape} (Subjects x ROIs)")
        return roi_df
    except Exception as e:
        print(f"  [Warning] ROI extraction using Harvard-Oxford atlas failed: {e}")
        return None


def generate_loso_splits(subjects):
    """
    Generates Leave-One-Subject-Out (LOSO) cross-validation split metadata.
    
    Returns:
        loso_splits (list of dict): List containing train/test subject lists per fold.
    """
    loso_splits = []
    n_subjects = len(subjects)
    
    for i, test_sub in enumerate(subjects):
        train_subs = [s for s in subjects if s != test_sub]
        loso_splits.append({
            "fold": i + 1,
            "test_subject": test_sub,
            "test_indices": [i],
            "train_subjects": train_subs,
            "train_indices": [j for j in range(n_subjects) if j != i]
        })
        
    print(f"\n🔄 Generated {len(loso_splits)} Leave-One-Subject-Out (LOSO) cross-validation folds.")
    return loso_splits


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction for ML and Deep Learning")
    parser.add_argument("--fsl_preproc_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/fsl", help="Path to preprocessing derivatives")
    parser.add_argument("--glm_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/glm", help="Path to GLM analysis outputs")
    parser.add_argument("--output_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/features", help="Path to output feature matrices")
    parser.add_argument("--pca_components", type=int, default=20, help="Number of PCA components to extract")
    args = parser.parse_args()

    fsl_preproc_dir = Path(args.fsl_preproc_dir)
    glm_dir = Path(args.glm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QC-passed subject list
    qc_file = fsl_preproc_dir / "qc_passed_subjects.json"
    if not qc_file.exists():
        print(f"ERROR: QC-passed subjects file not found at {qc_file}", file=sys.stderr)
        sys.exit(1)

    with open(qc_file, "r") as f:
        qc_data = json.load(f)
    qc_subjects = sorted(list(qc_data.keys()))
    print(f"Loaded {len(qc_subjects)} QC-passed subjects from {qc_file.name}")

    # 1. Load contrast maps (both zstat and cope)
    contrasts = [
        "IncongruentGreaterThanCongruent",
        "CongruentGreaterThanIncongruent"
    ]
    
    feature_metadata = {
        "qc_subjects": qc_subjects,
        "n_subjects": len(qc_subjects),
        "contrasts": contrasts
    }

    # Extract features for primary contrast: IncongruentGreaterThanCongruent
    primary_contrast = "IncongruentGreaterThanCongruent"
    
    # Load z-stat contrast maps
    zstat_paths, avail_subs = load_subject_contrast_maps(qc_subjects, glm_dir, primary_contrast, "zstat")
    # Load cope contrast maps
    cope_paths, _ = load_subject_contrast_maps(qc_subjects, glm_dir, primary_contrast, "cope")

    if not zstat_paths:
        print("ERROR: No contrast maps found for feature extraction.", file=sys.stderr)
        sys.exit(1)

    # 2. Whole-brain voxel feature extraction (zstat & cope)
    voxel_zstat, _ = extract_voxel_features(zstat_paths, MNI_MASK)
    voxel_cope, _ = extract_voxel_features(cope_paths, MNI_MASK)

    # Save voxel arrays
    np.save(output_dir / "features_voxel_zstat.npy", voxel_zstat)
    np.save(output_dir / "features_voxel_cope.npy", voxel_cope)

    # 3. PCA Feature Extraction
    pca_matrix, pca_obj = extract_pca_features(voxel_zstat, n_components=args.pca_components)
    np.save(output_dir / "features_pca.npy", pca_matrix)
    np.save(output_dir / "pca_explained_variance.npy", pca_obj.explained_variance_ratio_)

    # 4. ROI Feature Extraction
    roi_df = extract_roi_features(zstat_paths, avail_subs)
    if roi_df is not None:
        roi_df.to_csv(output_dir / "features_roi.csv")
        np.save(output_dir / "features_roi.npy", roi_df.values)

    # 5. Build dual-condition classification dataset (Incongruent vs. Congruent)
    # Stack condition 0: CongruentGreaterThanIncongruent, condition 1: IncongruentGreaterThanCongruent
    cog_paths, _ = load_subject_contrast_maps(qc_subjects, glm_dir, "CongruentGreaterThanIncongruent", "zstat")
    inc_paths, _ = load_subject_contrast_maps(qc_subjects, glm_dir, "IncongruentGreaterThanCongruent", "zstat")

    if cog_paths and inc_paths:
        cog_voxels, _ = extract_voxel_features(cog_paths, MNI_MASK)
        inc_voxels, _ = extract_voxel_features(inc_paths, MNI_MASK)

        X_classification = np.vstack([cog_voxels, inc_voxels])
        y_classification = np.array([0] * len(cog_paths) + [1] * len(inc_paths))
        subject_classification_ids = avail_subs + avail_subs

        np.save(output_dir / "X_classification_voxel.npy", X_classification)
        np.save(output_dir / "y_classification.npy", y_classification)
        
        # Also compute PCA for combined classification matrix
        pca_cls, _ = extract_pca_features(X_classification, n_components=args.pca_components)
        np.save(output_dir / "X_classification_pca.npy", pca_cls)

    # 6. Generate Leave-One-Subject-Out (LOSO) splits
    loso_splits = generate_loso_splits(avail_subs)
    with open(output_dir / "loso_splits.json", "w") as f:
        json.dump(loso_splits, f, indent=2)

    # Save summary metadata
    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=2)

    print("\n=======================================================")
    print("✅ Feature Extraction Completed Successfully!")
    print(f"Outputs saved to: {output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
