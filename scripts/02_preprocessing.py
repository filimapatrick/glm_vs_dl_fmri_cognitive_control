#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool

# Locate FSLDIR from environment
FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:
    # Try common default locations
    default_fsl = Path("/Users/patrickfilima/fsl")
    if default_fsl.exists():
        os.environ["FSLDIR"] = str(default_fsl)
        os.environ["PATH"] = f"{default_fsl}/bin:" + os.environ["PATH"]
        FSLDIR = str(default_fsl)
    else:
        print("ERROR: FSLDIR environment variable not found. Please install FSL and set FSLDIR.", file=sys.stderr)
        sys.exit(1)

# Ensure FSL binaries are in PATH
fsl_bin = Path(FSLDIR) / "bin"
if str(fsl_bin) not in os.environ["PATH"]:
    os.environ["PATH"] = f"{fsl_bin}:" + os.environ["PATH"]

# Path to MNI template
MNI_TEMPLATE = Path(FSLDIR) / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
if not MNI_TEMPLATE.exists():
    print(f"ERROR: MNI152 template not found at {MNI_TEMPLATE}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, description=""):
    """Run shell command and check for success."""
    if description:
        print(f"  [Running] {description}...")
    try:
        # Run command and capture output
        res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Command failed: {cmd}", file=sys.stderr)
        print(f"  [Stderr] {e.stderr}", file=sys.stderr)
        raise e


def preprocess_subject(subject_id, dataset_dir, output_dir, force=False):
    """Preprocess a single subject's functional and structural fMRI scans."""
    print(f"=== Preprocessing {subject_id} ===")
    
    # Define directories
    sub_raw_dir = Path(dataset_dir) / subject_id
    sub_out_dir = Path(output_dir) / subject_id
    
    anat_out_dir = sub_out_dir / "anat"
    func_out_dir = sub_out_dir / "func"
    tmp_dir = sub_out_dir / "tmp"
    
    # Check if raw files exist
    t1w_raw = sub_raw_dir / "anat" / f"{subject_id}_T1w.nii.gz"
    if not t1w_raw.exists():
        print(f"  [Skip] Raw T1w anatomical scan not found for {subject_id} at {t1w_raw}", file=sys.stderr)
        return False
        
    runs = [1, 2]
    valid_runs = []
    for run in runs:
        bold_raw = sub_raw_dir / "func" / f"{subject_id}_task-flanker_run-{run}_bold.nii.gz"
        if bold_raw.exists():
            valid_runs.append((run, bold_raw))
        else:
            print(f"  [Info] Run {run} BOLD scan not found for {subject_id}")
            
    if not valid_runs:
        print(f"  [Skip] No functional runs found for {subject_id}", file=sys.stderr)
        return False
        
    # Check if final preprocessed runs already exist to skip subject
    all_runs_preprocessed = True
    for run, _ in valid_runs:
        final_bold = func_out_dir / f"{subject_id}_task-flanker_run-{run}_space-MNI152_desc-preproc.nii.gz"
        if not final_bold.exists() or force:
            all_runs_preprocessed = False
            break
    
    if all_runs_preprocessed and not force:
        print(f"  [Skip] Subject {subject_id} already preprocessed. Use --force to rerun.")
        return True

    # Re-create output directories
    anat_out_dir.mkdir(parents=True, exist_ok=True)
    func_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Structural Brain Extraction (BET)
        t1w_brain = anat_out_dir / f"{subject_id}_T1w_brain.nii.gz"
        if not t1w_brain.exists() or force:
            run_cmd(
                f"bet {t1w_raw} {t1w_brain} -R -f 0.5 -g 0",
                description="BET (Structural Brain Extraction)"
            )
        else:
            print("  [BET] Output brain already exists.")

        # Step 2: Register Structural Brain to MNI152 Standard Space (12 DOF Affine)
        struct2mni_img = anat_out_dir / f"{subject_id}_T1w_space-MNI152.nii.gz"
        struct2mni_mat = anat_out_dir / f"{subject_id}_T1w_to_MNI152.mat"
        if not struct2mni_mat.exists() or force:
            run_cmd(
                f"flirt -in {t1w_brain} -ref {MNI_TEMPLATE} -out {struct2mni_img} -omat {struct2mni_mat} -dof 12 -cost corratio",
                description="FLIRT (Structural to MNI152 Registration, 12 DOF)"
            )
        else:
            print("  [Structural FLIRT] Registration matrix already exists.")

        # Preprocess each functional run
        for run, bold_raw in valid_runs:
            print(f"  --- Preprocessing Run {run} ---")
            
            # Temporary files
            bold_mcf = tmp_dir / f"run-{run}_bold_mcf.nii.gz"
            bold_st = tmp_dir / f"run-{run}_bold_st.nii.gz"
            bold_smooth = tmp_dir / f"run-{run}_bold_smooth.nii.gz"
            bold_filt = tmp_dir / f"run-{run}_bold_filt.nii.gz"
            bold_mean = tmp_dir / f"run-{run}_bold_mean.nii.gz"
            example_func = tmp_dir / f"run-{run}_example_func.nii.gz"
            
            # Final output files
            final_bold = func_out_dir / f"{subject_id}_task-flanker_run-{run}_space-MNI152_desc-preproc.nii.gz"
            motion_par_dest = func_out_dir / f"{subject_id}_task-flanker_run-{run}_motion.par"
            bold2struct_mat = func_out_dir / f"{subject_id}_task-flanker_run-{run}_to_T1w.mat"
            bold2mni_mat = func_out_dir / f"{subject_id}_task-flanker_run-{run}_to_MNI152.mat"

            # 2.1 Motion Correction (MCFLIRT)
            # Realign BOLD to the middle volume (73rd volume since TR is 2.0s, 146 volumes total)
            if not bold_mcf.exists() or force:
                run_cmd(
                    f"mcflirt -in {bold_raw} -out {bold_mcf} -refvol 73 -mats -plots -rmsrel -rmsabs",
                    description=f"MCFLIRT Run {run}"
                )
                # Copy the motion parameters file to the final destination
                motion_par_src = tmp_dir / f"run-{run}_bold_mcf.nii.gz.par"
                if motion_par_src.exists():
                    shutil.copy(motion_par_src, motion_par_dest)
            else:
                print(f"  [MCFLIRT] Run {run} outputs already exist.")

            # 2.2 Slice Timing Correction (Siemens interleaved slices)
            if not bold_st.exists() or force:
                run_cmd(
                    f"slicetimer -i {bold_mcf} -o {bold_st} -r 2.0",
                    description=f"Slicetimer Run {run}"
                )
            else:
                print(f"  [Slicetimer] Run {run} output already exists.")

            # 2.3 Spatial Smoothing (FWHM = 5.0mm -> sigma = 5 / 2.3548 = 2.123mm)
            if not bold_smooth.exists() or force:
                run_cmd(
                    f"fslmaths {bold_st} -s 2.123 {bold_smooth}",
                    description=f"Spatial Smoothing Run {run} (FWHM=5mm)"
                )
            else:
                print(f"  [Smoothing] Run {run} output already exists.")

            # 2.4 High-Pass Temporal Filtering (100s cutoff -> hp_sigma = 100s / (2 * TR) = 25 volumes)
            if not bold_filt.exists() or force:
                # To maintain signal intensity scale, we compute the temporal mean,
                # run high-pass filter, and add the mean back.
                print(f"  [Filtering] High-Pass Temporal Filter (100s) Run {run}...")
                run_cmd(f"fslmaths {bold_smooth} -Tmean {bold_mean}")
                run_cmd(f"fslmaths {bold_smooth} -bptf 25 -1 {bold_filt}")
                run_cmd(f"fslmaths {bold_filt} -add {bold_mean} {bold_filt}")
            else:
                print(f"  [Filtering] Run {run} output already exists.")

            # 2.5 Extract example BOLD volume for Registration reference (e.g. 73rd volume)
            if not example_func.exists() or force:
                run_cmd(f"fslroi {bold_mcf} {example_func} 73 1")

            # 2.6 Registration BOLD to T1w Structural (6 DOF Linear)
            if not bold2struct_mat.exists() or force:
                run_cmd(
                    f"flirt -in {example_func} -ref {t1w_brain} -omat {bold2struct_mat} -dof 6 -cost corratio",
                    description=f"FLIRT (BOLD to Structural Run {run})"
                )
            else:
                print(f"  [BOLD to Struct FLIRT] Mat already exists.")

            # 2.7 Concatenate matrices (BOLD -> T1w -> MNI152)
            if not bold2mni_mat.exists() or force:
                run_cmd(
                    f"convert_xfm -omat {bold2mni_mat} -concat {struct2mni_mat} {bold2struct_mat}",
                    description=f"Matrix Concatenation Run {run}"
                )
            else:
                print(f"  [XFM Concatenation] Combined mat already exists.")

            # 2.8 Apply combined matrix to 4D preprocessed BOLD data
            if not final_bold.exists() or force:
                run_cmd(
                    f"applyxfm4D {bold_filt} {MNI_TEMPLATE} {final_bold} {bold2mni_mat} -singlematrix",
                    description=f"Applying registration to BOLD standard space Run {run}"
                )
            else:
                print(f"  [Apply registration] Standard space BOLD already exists.")

        # Clean up tmp directory to save disk space
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"=== Successfully completed preprocessing for {subject_id} ===")
        return True

    except Exception as e:
        print(f"=== FAILED preprocessing for {subject_id}: {str(e)} ===", file=sys.stderr)
        # Keep tmp directory for debugging in case of failure
        return False


def main():
    parser = argparse.ArgumentParser(description="FSL Preprocessing Pipeline Wrapper for GLM-vs-DL")
    parser.add_argument("--dataset_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control", help="Path to raw BIDS dataset")
    parser.add_argument("--output_dir", default="/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/derivatives/fsl", help="Path to output preprocessed data")
    parser.add_argument("--subjects", default="all", help="Comma-separated list of subjects, or 'all'")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes to use (default: 1)")
    parser.add_argument("--force", action="store_true", help="Force recalculation of already processed subjects")
    args = parser.parse_args()

    # Determine subjects list
    dataset_path = Path(args.dataset_dir)
    if args.subjects == "all":
        # Find all sub-XX directories
        subjects = sorted([d.name for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print(f"Found {len(subjects)} subjects to process: {', '.join(subjects)}")
    
    # Run preprocessing
    if args.parallel > 1:
        print(f"Running in parallel with {args.parallel} jobs.")
        # Pre-wrap arguments
        tasks = [(sub, args.dataset_dir, args.output_dir, args.force) for sub in subjects]
        with Pool(args.parallel) as pool:
            results = pool.starmap(preprocess_subject, tasks)
        success_count = sum(1 for r in results if r)
    else:
        success_count = 0
        for sub in subjects:
            if preprocess_subject(sub, args.dataset_dir, args.output_dir, args.force):
                success_count += 1

    print(f"\nPreprocessing finished. Successfully processed {success_count} / {len(subjects)} subjects.")


if __name__ == "__main__":
    main()
