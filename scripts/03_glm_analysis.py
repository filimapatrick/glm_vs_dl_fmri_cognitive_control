#!/usr/bin/env python3
"""
scripts/03_glm_analysis.py - Classical GLM Analysis
Performs first-level (subject-level) fMRI GLM analysis using Nilearn
and group-level mixed-effects modeling using FSL FLAME 1+2.
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import pandas as pd
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs

# Locate FSLDIR from environment or default
FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:
    default_fsl = Path("/Users/patrickfilima/fsl")
    if default_fsl.exists():
        os.environ["FSLDIR"] = str(default_fsl)
        os.environ["PATH"] = f"{default_fsl}/bin:" + os.environ["PATH"]
        FSLDIR = str(default_fsl)
    else:
        print("ERROR: FSLDIR environment variable not found. Please install FSL.", file=sys.stderr)
        sys.exit(1)

# Ensure FSL binaries are in PATH
fsl_bin = Path(FSLDIR) / "bin"
if str(fsl_bin) not in os.environ["PATH"]:
    os.environ["PATH"] = f"{fsl_bin}:" + os.environ["PATH"]

# standard space brain mask for MNI152 2mm
MNI_MASK = Path(FSLDIR) / "data" / "standard" / "MNI152_T1_2mm_brain_mask.nii.gz"
if not MNI_MASK.exists():
    print(f"ERROR: MNI152 brain mask not found at {MNI_MASK}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, description=""):
    """Run shell command and check for success."""
    if description:
        print(f"  [Running] {description}...")
    try:
        res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Command failed: {cmd}", file=sys.stderr)
        print(f"  [Stderr] {e.stderr}", file=sys.stderr)
        raise e


def parse_smoothest_output(output_str):
    """Parse output from FSL smoothest command to extract DLH and VOLUME."""
    dlh = None
    volume = None
    for line in output_str.strip().split('\n'):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "DLH":
            dlh = float(parts[1])
        elif parts[0] == "VOLUME":
            volume = int(parts[1])
    return dlh, volume


def write_fsl_design_files(num_subjects, out_dir):
    """Generate FSL format design matrix, contrast, and group split files for a 1-sample t-test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mat_file = out_dir / "design.mat"
    con_file = out_dir / "design.con"
    grp_file = out_dir / "design.grp"
    
    # Write design.mat
    with open(mat_file, "w") as f:
        f.write(f"/NumWaves 1\n")
        f.write(f"/NumPoints {num_subjects}\n")
        f.write(f"/PPheights 1.000000e+00\n\n")
        f.write(f"/Matrix\n")
        for _ in range(num_subjects):
            f.write(f"1.000000e+00\n")
            
    # Write design.con
    with open(con_file, "w") as f:
        f.write(f"/ContrastName1 group_mean\n")
        f.write(f"/NumWaves 1\n")
        f.write(f"/NumContrasts 1\n")
        f.write(f"/PPheights 1.000000e+00\n")
        f.write(f"/RequiredEffect1 1.000\n\n")
        f.write(f"/Matrix\n")
        f.write(f"1.000000e+00\n")
        
    # Write design.grp
    with open(grp_file, "w") as f:
        f.write(f"/NumWaves 1\n")
        f.write(f"/NumPoints {num_subjects}\n\n")
        f.write(f"/Matrix\n")
        for _ in range(num_subjects):
            f.write(f"1\n")
            
    return mat_file, con_file, grp_file


def run_first_level(subject_id, runs, dataset_dir, fsl_preproc_dir, output_dir, force=False):
    """Run subject-level GLM using Nilearn and save COPE, VARCOPE, and Zstat maps."""
    sub_out_dir = Path(output_dir) / subject_id
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    
    contrasts = {
        "IncongruentGreaterThanCongruent": "incongruent - congruent",
        "CongruentGreaterThanIncongruent": "congruent - incongruent"
    }
    
    # Check if outputs already exist to skip
    all_outputs_exist = True
    for contrast_name in contrasts.keys():
        for map_type in ["cope", "varcope", "zstat"]:
            out_file = sub_out_dir / f"{subject_id}_contrast-{contrast_name}_{map_type}.nii.gz"
            if not out_file.exists() or force:
                all_outputs_exist = False
                break
    
    if all_outputs_exist:
        print(f"  [Skip] First-level GLM outputs already exist for {subject_id}.")
        return True

    print(f"--- First-level GLM for {subject_id} (Runs: {runs}) ---")
    
    run_imgs = []
    run_events = []
    run_confounds = []
    
    for run in runs:
        # Define file paths
        bold_file = Path(fsl_preproc_dir) / subject_id / "func" / f"{subject_id}_task-flanker_{run}_space-MNI152_desc-preproc.nii.gz"
        events_file = Path(dataset_dir) / subject_id / "func" / f"{subject_id}_task-flanker_{run}_events.tsv"
        motion_file = Path(fsl_preproc_dir) / subject_id / "func" / f"{subject_id}_task-flanker_{run}_motion.par"
        
        # Verify existence
        if not bold_file.exists():
            print(f"  [Error] Preprocessed BOLD not found: {bold_file}", file=sys.stderr)
            return False
        if not events_file.exists():
            print(f"  [Error] Events file not found: {events_file}", file=sys.stderr)
            return False
        if not motion_file.exists():
            print(f"  [Error] Motion par file not found: {motion_file}", file=sys.stderr)
            return False
            
        # Load and clean events
        events_df = pd.read_csv(events_file, sep='\t')
        # Map trial_type to Stimulus so all trials are included and balanced
        nilearn_events = pd.DataFrame({
            'onset': events_df['onset'],
            'duration': events_df['duration'],
            'trial_type': events_df['Stimulus'] # congruent / incongruent
        })
        
        # Load motion confound regressors
        motion_df = pd.read_csv(motion_file, sep=r'\s+', header=None)
        motion_df.columns = [f'motion_{i}' for i in range(6)]
        
        run_imgs.append(str(bold_file))
        run_events.append(nilearn_events)
        run_confounds.append(motion_df)
        
    try:
        # Fit GLM model across subject runs
        # hrf_model='glover', drift_model=None (already temporally filtered in preprocessing)
        model = FirstLevelModel(
            t_r=2.0,
            hrf_model='glover',
            drift_model=None,
            mask_img=str(MNI_MASK),
            verbose=0
        )
        model.fit(run_imgs, events=run_events, confounds=run_confounds)
        
        # Compute and save contrasts
        for contrast_name, contrast_def in contrasts.items():
            print(f"  Computing contrast: {contrast_name}")
            
            # COPE (Contrast Parameter Estimate / Effect Size)
            cope_map = model.compute_contrast(contrast_def, output_type='effect_size')
            cope_map.to_filename(str(sub_out_dir / f"{subject_id}_contrast-{contrast_name}_cope.nii.gz"))
            
            # VARCOPE (Variance of Contrast Parameter Estimate)
            varcope_map = model.compute_contrast(contrast_def, output_type='effect_variance')
            varcope_map.to_filename(str(sub_out_dir / f"{subject_id}_contrast-{contrast_name}_varcope.nii.gz"))
            
            # Z-statistic
            zstat_map = model.compute_contrast(contrast_def, output_type='z_score')
            zstat_map.to_filename(str(sub_out_dir / f"{subject_id}_contrast-{contrast_name}_zstat.nii.gz"))
            
        print(f"  Successfully finished first-level GLM for {subject_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Subject {subject_id} failed first-level GLM: {str(e)}", file=sys.stderr)
        return False


def run_group_level(contrast_name, qc_subjects, output_dir, force=False):
    """Concatenate subject maps, generate FSL design files, run FLAME 1+2, and perform cluster correction."""
    group_out_dir = Path(output_dir) / "group"
    group_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if final group outputs already exist to skip
    final_thresh_z = group_out_dir / f"group_contrast-{contrast_name}_thresh_zstat.nii.gz"
    final_zstat = group_out_dir / f"group_contrast-{contrast_name}_zstat.nii.gz"
    final_cope = group_out_dir / f"group_contrast-{contrast_name}_cope.nii.gz"
    final_varcope = group_out_dir / f"group_contrast-{contrast_name}_varcope.nii.gz"
    final_cluster_table = group_out_dir / f"group_contrast-{contrast_name}_cluster_table.txt"
    
    if final_thresh_z.exists() and final_zstat.exists() and final_cope.exists() and final_varcope.exists() and final_cluster_table.exists() and not force:
        print(f"\n[Skip] Group-level analysis for {contrast_name} already exists.")
        return True
        
    print(f"\n==========================================")
    print(f"Group-level Analysis: {contrast_name}")
    print(f"==========================================")
    
    copes_list = []
    varcopes_list = []
    
    # Collect all subject files
    for subject_id in qc_subjects:
        sub_cope = Path(output_dir) / subject_id / f"{subject_id}_contrast-{contrast_name}_cope.nii.gz"
        sub_varcope = Path(output_dir) / subject_id / f"{subject_id}_contrast-{contrast_name}_varcope.nii.gz"
        
        if not sub_cope.exists() or not sub_varcope.exists():
            print(f"  [Warning] Missing first-level GLM outputs for {subject_id}. Skipping from group level.", file=sys.stderr)
            continue
            
        copes_list.append(sub_cope)
        varcopes_list.append(sub_varcope)
        
    num_subs = len(copes_list)
    print(f"  Found first-level data for {num_subs} subjects.")
    if num_subs < 2:
        print("  [Error] Insufficient subjects for group-level analysis.", file=sys.stderr)
        return False
        
    # Temporary files directory for flameo
    tmp_flame_dir = group_out_dir / f"{contrast_name}_flame_work"
    tmp_flame_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Concatenate subject-level maps into 4D NIfTIs
        print("  Concatenating subject-level COPEs and VARCOPEs into 4D images...")
        copes_4d_file = tmp_flame_dir / "copes_4d.nii.gz"
        varcopes_4d_file = tmp_flame_dir / "varcopes_4d.nii.gz"
        
        concat_imgs(copes_list).to_filename(str(copes_4d_file))
        concat_imgs(varcopes_list).to_filename(str(varcopes_4d_file))
        
        # Step 2: Write design matrix and contrasts
        print("  Generating FSL design matrix and contrast files...")
        mat_file, con_file, grp_file = write_fsl_design_files(num_subs, tmp_flame_dir)
        
        # Step 3: Run FSL flameo for Mixed-effects (FLAME 1+2)
        flame_log_dir = tmp_flame_dir / "flame_output"
        if flame_log_dir.exists():
            shutil.rmtree(flame_log_dir)
        
        print("  Running FSL flameo (FLAME 1+2)...")
        flame_cmd = (
            f"flameo --cope={copes_4d_file} "
            f"--varcope={varcopes_4d_file} "
            f"--mask={MNI_MASK} "
            f"--dm={mat_file} "
            f"--tc={con_file} "
            f"--cs={grp_file} "
            f"--runmode=flame12 "
            f"--ld={flame_log_dir}"
        )
        run_cmd(flame_cmd, description="FSL FLAME 1+2")
        
        flame_zstat = flame_log_dir / "zstat1.nii.gz"
        flame_cope = flame_log_dir / "cope1.nii.gz"
        flame_varcope = flame_log_dir / "varcope1.nii.gz"
        
        if not flame_zstat.exists():
            print("  [Error] flameo output zstat1.nii.gz not found!", file=sys.stderr)
            return False
            
        # Copy raw group results to final destination
        shutil.copy(flame_zstat, final_zstat)
        shutil.copy(flame_cope, final_cope)
        shutil.copy(flame_varcope, final_varcope)
        
        # Step 4: Estimate spatial smoothness of Z-stat map using FSL smoothest
        print("  Estimating spatial smoothness using FSL smoothest...")
        smooth_cmd = f"smoothest -z {final_zstat} -m {MNI_MASK}"
        smooth_out = run_cmd(smooth_cmd, description="FSL smoothest")
        dlh, vol = parse_smoothest_output(smooth_out)
        print(f"    DLH: {dlh:.6f}, Volume in voxels: {vol}")
        
        # Step 5: Perform multiple comparisons correction using FSL fsl-cluster
        # Threshold: Z > 3.1, p < 0.05 corrected
        print("  Applying cluster-based thresholding (Z > 3.1, p < 0.05 corrected)...")
        cluster_cmd = (
            f"fsl-cluster --in={final_zstat} "
            f"--thresh=3.1 "
            f"--pthresh=0.05 "
            f"--dlh={dlh} "
            f"--volume={vol} "
            f"--cope={final_cope} "
            f"--othresh={final_thresh_z} "
            f"--oindex={group_out_dir / f'group_contrast-{contrast_name}_cluster_index.nii.gz'} "
            f"--olmax={group_out_dir / f'group_contrast-{contrast_name}_lmax.txt'} "
            f"> {final_cluster_table}"
        )
        run_cmd(cluster_cmd, description="FSL fsl-cluster")
        
        # Display the cluster table output
        print("\n📈 Cluster Summary Table:")
        if final_cluster_table.exists():
            with open(final_cluster_table, "r") as f:
                print(f.read())
                
        # Clean up temporary flame workspace
        shutil.rmtree(tmp_flame_dir, ignore_errors=True)
        print(f"Group-level analysis for contrast {contrast_name} completed successfully.")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Group-level analysis failed: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="FSL Classical GLM Analysis Pipeline (Nilearn + flameo)")
    parser.add_argument("--dataset_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control", help="Path to raw BIDS dataset")
    parser.add_argument("--fsl_preproc_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/fsl", help="Path to preprocessed data")
    parser.add_argument("--output_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/glm", help="Path to output GLM results")
    parser.add_argument("--force", action="store_true", help="Force recalculation of existing outputs")
    args = parser.parse_args()
    
    # Load QC-passed subjects
    qc_file = Path(args.fsl_preproc_dir) / "qc_passed_subjects.json"
    if not qc_file.exists():
        print(f"ERROR: QC-passed subjects JSON file not found at {qc_file}", file=sys.stderr)
        print("Please run scripts/01_quality_control.py and preprocessing first.", file=sys.stderr)
        sys.exit(1)
        
    with open(qc_file, "r") as f:
        qc_data = json.load(f)
        
    # Perform first-level analysis for each QC-passed subject
    print(f"Found {len(qc_data)} QC-passed subjects in {qc_file.name}")
    
    success_subjects = []
    for subject_id, runs in qc_data.items():
        if run_first_level(subject_id, runs, args.dataset_dir, args.fsl_preproc_dir, args.output_dir, args.force):
            success_subjects.append(subject_id)
            
    print(f"\nFirst-level GLM finished. Successful subjects: {len(success_subjects)} / {len(qc_data)}")
    
    # Perform group-level analysis
    contrasts = [
        "IncongruentGreaterThanCongruent",
        "CongruentGreaterThanIncongruent"
    ]
    
    group_success = 0
    for contrast_name in contrasts:
        if run_group_level(contrast_name, success_subjects, args.output_dir, args.force):
            group_success += 1
            
    print(f"\n==========================================")
    print(f"GLM Analysis Pipeline Completed.")
    print(f"Group level contrasts processed: {group_success} / {len(contrasts)}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
