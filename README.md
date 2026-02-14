# A Comparative Study of General Linear Modeling and Deep Learning Approaches for Task-Based fMRI Analysis of Cognitive Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FSL](https://img.shields.io/badge/FSL-6.0+-green.svg)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

## 📋 Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contributors](#contributors)

## 🔬 Overview

This project provides a systematic comparison between classical General Linear Model (GLM)-based analysis and modern deep learning approaches for analyzing task-based functional MRI data from the Eriksen Flanker task. We investigate cognitive control mechanisms using the well-established ds000102 dataset from OpenNeuro.

### Key Research Questions
- Do deep learning methods offer meaningful advantages over traditional GLM approaches in small-scale fMRI datasets?
- How do performance, sensitivity, and interpretability compare between these methodological approaches?
- What are the optimal strategies for applying deep learning to task-based fMRI with limited sample sizes?

## 🎯 Objectives

1. **Preprocessing**: Apply standardized FSL preprocessing pipelines to task-based fMRI data
2. **GLM Analysis**: Identify cognitive control brain regions using classical statistical approaches
3. **Feature Extraction**: Develop robust feature extraction methods for machine learning applications
4. **Deep Learning**: Design and implement neural networks for cognitive state classification
5. **Comparison**: Systematically evaluate and compare methodological approaches
6. **Validation**: Assess generalizability and practical utility for cognitive neuroscience research

## 📊 Dataset

### NYU Slow Flanker Dataset (ds000102)
- **Source**: [OpenNeuro](https://openneuro.org/datasets/ds000102)
- **Participants**: 26 healthy adults (ages 19-50)
- **Task**: Eriksen Flanker task (cognitive control paradigm)
- **Design**: Event-related, 2 conditions (congruent vs incongruent)
- **Acquisition**: 3T Siemens Allegra, TR=2s, 146 volumes per run

### Task Description
Participants indicate the direction of a central arrow in a 5-arrow array:
- **Congruent trials**: Flanking arrows point same direction (e.g., `< < < < <`)
- **Incongruent trials**: Flanking arrows point opposite direction (e.g., `< < > < <`)
- **Timing**: 8-14s ITI (mean=12s), 24 trials per condition per subject

### Data Structure
```
├── sub-XX/
│   ├── anat/
│   │   └── sub-XX_T1w.nii.gz          # Anatomical scan
│   └── func/
│       ├── sub-XX_task-flanker_run-1_bold.nii.gz    # Functional run 1
│       ├── sub-XX_task-flanker_run-1_events.tsv     # Behavioral data 1
│       ├── sub-XX_task-flanker_run-2_bold.nii.gz    # Functional run 2
│       └── sub-XX_task-flanker_run-2_events.tsv     # Behavioral data 2
├── derivatives/mriqc/                  # Quality control metrics
├── dataset_description.json
├── participants.tsv
└── task-flanker_bold.json
```

## 🔄 Methodology

### Phase 1: Data Preparation & Quality Control
```bash
# Quality assessment using MRIQC metrics
- Motion parameters (mean FD, DVARS)
- Signal quality (tSNR, temporal SNR)  
- Artifact detection and exclusion criteria
```

### Phase 2: FSL-Based Preprocessing
```bash
# Standard FSL pipeline
1. Brain extraction (BET)
2. Motion correction (MCFLIRT)  
3. Slice timing correction
4. Spatial smoothing (FWHM=5mm)
5. High-pass temporal filtering (100s cutoff)
6. Registration to MNI152 standard space
```

### Phase 3: Classical GLM Analysis
```python
# First-level analysis
- Model: Canonical HRF convolution
- Regressors: Congruent, Incongruent, Motion parameters
- Contrasts: [Incongruent > Congruent], [Congruent > Incongruent]

# Group-level analysis  
- Mixed-effects modeling (FLAME 1+2)
- Multiple comparisons correction (cluster-based thresholding)
- Statistical maps: Z > 3.1, p < 0.05 corrected
```

### Phase 4: Feature Extraction for Machine Learning
```python
# Feature types
1. Voxel-wise contrasts (whole-brain or masked)
2. ROI-based activation summaries 
3. Connectivity features (seed-based correlation)
4. Dimensionality reduction (PCA, ICA components)

# Cross-validation strategy
- Leave-one-subject-out (LOSO)
- Stratified k-fold within subjects
- Proper train/validation/test splits
```

### Phase 5: Deep Learning Implementation
```python
# Architecture considerations
- Shallow networks to prevent overfitting
- Convolutional layers for spatial features
- Dropout and L2 regularization
- Batch normalization

# Model variants
1. Multi-layer Perceptron (MLP) on extracted features  
2. 1D CNN on time series
3. 3D CNN on activation maps (with heavy regularization)
4. Ensemble methods combining multiple approaches
```

### Phase 6: Comparative Evaluation
```python
# Performance metrics
- Classification accuracy, precision, recall, F1-score
- ROC-AUC and precision-recall curves  
- Cross-validation stability
- Statistical significance testing

# Interpretability analysis
- GLM activation maps vs DL saliency maps
- Feature importance rankings
- Spatial overlap quantification
```

## 📁 Project Structure

```
glm_vs_dl_fmri_cognitive_control/
├── data/                           # Raw BIDS dataset (ds000102)
├── derivatives/                    # Processed data outputs
│   ├── fsl/                       # FSL preprocessing results
│   ├── glm/                       # GLM analysis outputs  
│   ├── features/                  # Extracted ML features
│   └── models/                    # Trained DL models
├── scripts/
│   ├── 01_quality_control.py      # Data QC and exclusions
│   ├── 02_preprocessing.sh        # FSL preprocessing pipeline
│   ├── 03_glm_analysis.py         # Classical GLM analysis
│   ├── 04_feature_extraction.py   # ML feature preparation
│   ├── 05_deep_learning.py        # DL model training
│   └── 06_comparison_analysis.py  # Method comparison
├── notebooks/                     # Jupyter analysis notebooks
│   ├── exploratory_analysis.ipynb
│   ├── results_visualization.ipynb
│   └── supplementary_analyses.ipynb
├── results/                       # Output figures and tables
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies  
└── README.md                      # This file
```

## 🛠 Installation

### Prerequisites
- **FSL 6.0+** ([Installation guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation))
- **Python 3.8+** with scientific computing stack
- **Git LFS** for large file handling

### Environment Setup

#### Option 1: Conda (Recommended)
```bash
# Clone repository
git clone https://github.com/filimapatrick/glm_vs_dl_fmri_cognitive_control.git
cd glm_vs_dl_fmri_cognitive_control

# Create conda environment
conda env create -f environment.yml
conda activate fmri-glm-dl

# Verify FSL installation
echo $FSLDIR
fsl --version
```

#### Option 2: pip + virtualenv
```bash
# Create virtual environment
python -m venv fmri_env
source fmri_env/bin/activate  # Linux/Mac
# fmri_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
```yaml
# Core packages
- numpy >= 1.21.0
- scipy >= 1.7.0  
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- nibabel >= 3.2.0

# Deep learning
- tensorflow >= 2.8.0
- pytorch >= 1.10.0
- torchvision >= 0.11.0

# Neuroimaging
- nilearn >= 0.8.0
- nipype >= 1.7.0
- pybids >= 0.13.0

# Visualization
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
```

## 🚀 Usage

### Quick Start
```bash
# 1. Download dataset (if not already available)
python scripts/download_data.py --dataset ds000102 --output data/

# 2. Run complete analysis pipeline
bash run_analysis.sh

# 3. Generate results summary
python scripts/generate_report.py --output results/
```

### Step-by-Step Analysis
```bash
# Quality control and data preparation
python scripts/01_quality_control.py --input data/ --output derivatives/qc/

# FSL preprocessing  
bash scripts/02_preprocessing.sh --subjects all --output derivatives/fsl/

# GLM analysis
python scripts/03_glm_analysis.py --input derivatives/fsl/ --output derivatives/glm/

# Feature extraction for ML
python scripts/04_feature_extraction.py --glm derivatives/glm/ --output derivatives/features/

# Deep learning training and evaluation
python scripts/05_deep_learning.py --features derivatives/features/ --output derivatives/models/

# Comparative analysis
python scripts/06_comparison_analysis.py --glm derivatives/glm/ --models derivatives/models/ --output results/
```

### Configuration
Modify analysis parameters in `config.yaml`:
```yaml
# Preprocessing parameters
preprocessing:
  smoothing_fwhm: 5.0
  highpass_cutoff: 100.0
  
# GLM parameters  
glm:
  cluster_threshold: 3.1
  cluster_probability: 0.05
  
# Deep learning parameters
deep_learning:
  batch_size: 32
  learning_rate: 0.001
  dropout_rate: 0.5
  max_epochs: 200
```

## 📈 Expected Results

### GLM Analysis Outcomes
- **Activation maps**: Frontal-parietal control network engagement
- **Contrasts**: Incongruent > Congruent in ACC, IFG, IPS
- **Effect sizes**: Cohen's d typically 0.5-1.2 in key regions
- **Behavioral correlation**: RT differences correlate with brain activation

### Deep Learning Performance
- **Classification accuracy**: 65-80% (above chance level of 50%)
- **Cross-validation stability**: ±5-10% across folds  
- **Feature importance**: Frontal and parietal regions most predictive
- **Overfitting assessment**: Training vs validation learning curves

### Method Comparison
- **Spatial overlap**: 40-70% between GLM and DL important regions
- **Sensitivity analysis**: DL may detect subtle distributed patterns
- **Interpretability trade-offs**: GLM more interpretable, DL potentially more sensitive
- **Sample size effects**: Performance degradation analysis for smaller samples

## 📊 Quality Metrics

### Data Quality Thresholds
```python
# Inclusion criteria
max_mean_fd = 0.5      # Maximum mean framewise displacement (mm)
max_dvars = 75         # Maximum DVARS threshold  
min_tsnr = 50          # Minimum temporal signal-to-noise ratio
max_outlier_pct = 10   # Maximum percentage outlier volumes
```

### Model Performance Benchmarks
```python
# Expected performance ranges
GLM_sensitivity = [0.70, 0.85]      # Activation detection sensitivity
DL_accuracy = [0.65, 0.80]          # Classification accuracy  
cross_val_stability = 0.05          # Maximum CV standard deviation
spatial_overlap = [0.40, 0.70]      # GLM-DL spatial correlation
```

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@article{your_study_2026,
  title={A Comparative Study of General Linear Modeling and Deep Learning Approaches for Task-Based fMRI Analysis of Cognitive Control},
  author={Your Name and Colleagues},
  journal={Journal of Neuroinformatics},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={10.1007/sXXXXX-XXX-XXXX-X}
}
```

### Dataset Citation
```bibtex
@article{kelly2008competition,
  title={Competition between functional brain networks mediates behavioral variability},
  author={Kelly, AMC and Uddin, LQ and Biswal, BB and Castellanos, FX and Milham, MP},
  journal={NeuroImage},
  volume={39},
  number={1},
  pages={527--537},
  year={2008},
  publisher={Elsevier}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)  
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenNeuro** for providing the ds000102 dataset
- **FSL team** at Oxford for neuroimaging tools
- **Nilearn developers** for Python neuroimaging utilities
- **NYU** research team for original data collection

## 📞 Contact

- **Principal Investigator**: [Your Name](mailto:your.email@institution.edu)
- **GitHub Issues**: [Project Issues](https://github.com/yourusername/glm_vs_dl_fmri_cognitive_control/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/glm_vs_dl_fmri_cognitive_control/wiki)

---

**🧠 Advancing neuroimaging methodology through rigorous comparison of classical and modern analytical approaches.**# glm_vs_dl_fmri_cognitive_control
