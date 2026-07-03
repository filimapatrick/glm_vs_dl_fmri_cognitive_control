#!/usr/bin/env python3
import os
import sys
import csv
import json
from pathlib import Path

# Define paths
DATASET_DIR = Path("/Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control")
MRIQC_CSV = DATASET_DIR / "derivatives" / "mriqc" / "fMRIQC.csv"
OUTPUT_JSON = DATASET_DIR / "derivatives" / "fsl" / "qc_passed_subjects.json"

# Thresholds defined in the study design
MAX_MEAN_FD = 0.5      # Maximum mean Framewise Displacement (mm)
MAX_DVARS = 75.0       # Maximum DVARS threshold (if scaled as standard, otherwise informational)
MIN_TSNR = 40.0        # Minimum temporal Signal-to-Noise Ratio (tSNR)

def load_qc_metrics(csv_path):
    """Load fMRIQC metrics from CSV file using python built-in csv module."""
    if not csv_path.exists():
        print(f"ERROR: MRIQC CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
        
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze_distributions(data):
    """Analyze distributions of key metrics across the dataset."""
    fds = []
    tsnrs = []
    dvarss = []
    
    for row in data:
        try:
            fds.append(float(row['mean_fd']))
            tsnrs.append(float(row['m_tsnr']))
            dvarss.append(float(row['dvars']))
        except (ValueError, KeyError):
            continue
            
    if not fds:
        return
        
    print("\n📊 Dataset-wide Quality Metric Distributions:")
    print(f"  mean_fd:  min={min(fds):.4f}, max={max(fds):.4f}, mean={sum(fds)/len(fds):.4f}")
    print(f"  m_tsnr:   min={min(tsnrs):.2f}, max={max(tsnrs):.2f}, mean={sum(tsnrs)/len(tsnrs):.2f}")
    print(f"  dvars:    min={min(dvarss):.4f}, max={max(dvarss):.4f}, mean={sum(dvarss)/len(dvarss):.4f}")

def apply_qc_filters(data):
    """Apply QC filters to identify which runs/subjects pass or fail."""
    passed_runs = {}
    failed_details = []
    
    for row in data:
        sub = row['subject']
        # The scan name is typically like 'task-flanker_run-1' or similar
        scan = row['scan']
        run_name = "run-1" if "run-1" in scan else "run-2" if "run-2" in scan else scan
        
        try:
            fd = float(row['mean_fd'])
            tsnr = float(row['m_tsnr'])
            dvars = float(row['dvars'])
        except (ValueError, KeyError) as e:
            print(f"  [Warning] Missing metric for {sub} {scan}: {e}", file=sys.stderr)
            failed_details.append({
                'subject': sub, 'scan': scan, 'reason': 'Missing metrics'
            })
            continue
            
        # Check against criteria
        reasons = []
        if fd > MAX_MEAN_FD:
            reasons.append(f"FD={fd:.4f} > {MAX_MEAN_FD}")
        if tsnr < MIN_TSNR:
            reasons.append(f"tSNR={tsnr:.2f} < {MIN_TSNR}")
        if dvars > MAX_DVARS:
            reasons.append(f"DVARS={dvars:.2f} > {MAX_DVARS}")
            
        if not reasons:
            # Passed all QC checks
            if sub not in passed_runs:
                passed_runs[sub] = []
            passed_runs[sub].append(run_name)
        else:
            failed_details.append({
                'subject': sub,
                'scan': scan,
                'metrics': f"FD={fd:.4f}, tSNR={tsnr:.1f}, DVARS={dvars:.2f}",
                'reasons': ", ".join(reasons)
            })
            
    # Sort dictionaries and lists for reproducibility
    passed_runs = {k: sorted(v) for k, v in sorted(passed_runs.items())}
    return passed_runs, failed_details

def main():
    print("=== Phase 1: Data Preparation & Quality Control ===")
    data = load_qc_metrics(MRIQC_CSV)
    print(f"Loaded QC records for {len(data)} fMRI runs.")
    
    analyze_distributions(data)
    passed_runs, failed_details = apply_qc_filters(data)
    
    # Save the inclusion list to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(passed_runs, f, indent=4)
    print(f"\nSaved QC-passed inclusion list to: {OUTPUT_JSON}")
    
    # Print Excluded Runs Report
    print("\n❌ Excluded Runs Report:")
    if not failed_details:
        print("  None. All runs passed the QC filters.")
    else:
        print(f"  {'Subject':<10} | {'Scan':<20} | {'Metrics':<30} | {'Exclusion Reasons':<30}")
        print("  " + "-" * 103)
        for fail in failed_details:
            print(f"  {fail['subject']:<10} | {fail['scan']:<20} | {fail['metrics']:<30} | {fail['reasons']:<30}")
            
    # Summary stats
    total_subjects = len(set(row['subject'] for row in data))
    passed_subjects = len(passed_runs)
    total_runs = len(data)
    passed_runs_count = sum(len(runs) for runs in passed_runs.values())
    
    print("\n📊 Quality Control Summary Statistics:")
    print(f"  Total Subjects: {total_subjects} -> Passed QC: {passed_subjects} (Excluded: {total_subjects - passed_subjects})")
    print(f"  Total BOLD Runs: {total_runs} -> Passed QC: {passed_runs_count} (Excluded: {total_runs - passed_runs_count})")
    
    # Detail on subjects with partial data (e.g. only 1 run passed)
    partial_subjects = [sub for sub, runs in passed_runs.items() if len(runs) < 2]
    if partial_subjects:
        print(f"  Subjects with only 1 run passing QC: {', '.join(partial_subjects)}")
        
if __name__ == "__main__":
    main()
