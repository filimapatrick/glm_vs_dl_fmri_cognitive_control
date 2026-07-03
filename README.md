# A Methodological Study of GLM and Deep Learning Behavior in Small-Sample Task-Based fMRI of Cognitive Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FSL](https://img.shields.io/badge/FSL-6.0+-green.svg)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

## 📋 Table of Contents
- [Overview](#overview)
- [What is New in This Study](#what-is-new-in-this-study)
- [Objectives](#objectives)
- [Experimental Design](#experimental-design)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contributors](#contributors)

---

## 🔬 Overview

This project investigates model behavior under small-sample, high-dimensional fMRI conditions using the NYU Slow Flanker dataset (ds000102). We compare classical General Linear Models (GLM) and deep learning (DL) approaches to characterize differences in **predictive generalization, statistical stability, and spatial correspondence** under constrained data regimes typical of task-based neuroimaging studies.

Rather than treating this as a performance comparison, this study focuses on characterizing how different model classes behave under identical experimental constraints, with emphasis on **overfitting dynamics, cross-subject generalization, and representational alignment with task-evoked brain activity**.

The goal is to empirically characterize the limitations and behavior of deep learning in small-sample cognitive neuroscience, rather than to establish superiority over classical statistical inference.

### Key Research Questions
- **Generalization**: How do statistical inference (GLM) and predictive models (DL) differ in cross-subject generalization under small-sample neuroimaging constraints?
- **Stability**: How does statistical inference stability (GLM) compare to predictive stability (DL) under repeated cross-validation?
- **Spatial correspondence**: To what extent do DL-derived attribution maps correspond to GLM-derived activation maps?
- **Overfitting**: How do learning dynamics differ across model classes under controlled sample-size constraints?

---

## 💡 What is New in This Study

This work does not introduce new algorithms or propose new machine learning methods. Instead, it provides a **controlled methodological investigation** of how classical statistical inference (General Linear Modeling, GLM) and deep learning (DL) behave under realistic small-sample, high-dimensional fMRI constraints.

While the individual components of this pipeline (GLM analysis, deep learning classification, cross-validation, and permutation testing) are well established in neuroimaging and machine learning, their integration into a **single unified experimental framework focused on model behavior rather than performance optimization** is less commonly addressed.

### 1. Cross-Paradigm Comparison under a Shared Evaluation Framework
Most prior studies evaluate GLM as a statistical inference tool or DL/ML models as predictive classifiers independently. Here, both are evaluated under the **same dataset, preprocessing pipeline, and subject-level cross-validation protocol**, enabling a direct comparison between **statistical inference behavior and predictive modeling behavior** rather than isolated performance comparisons.

### 2. Focus on Small-Sample Instability Regimes (N=26)
Rather than treating small sample size as a limitation to be overcome, this study treats it as the primary experimental condition. We explicitly characterize how DL models behave under severe sample constraints, including:
- cross-validation variance
- instability across folds
- degradation under reduced training samples

The focus is on **failure modes and instability patterns**, not accuracy maximization.

### 3. Quantification of Overfitting as a Comparative Property
Unlike typical neuroimaging ML studies where overfitting is qualitative, here it is treated as a **quantifiable model property across model classes**. We measure:
- training–validation divergence
- learning-curve dynamics
- cross-validation variability

These are compared across GLM-based statistical stability and DL predictive instability under identical data partitions.

### 4. Spatial Correspondence Between Statistical and Learned Representations
We evaluate whether DL-derived spatial attribution patterns align with GLM-derived activation maps. This is treated as a **primary analysis objective rather than post-hoc interpretability**.

Spatial correspondence is quantified using:
- Pearson correlation of spatial maps
- Dice coefficient (thresholded overlap)

This allows evaluation of whether predictive models recover canonical cognitive control networks identified by GLM.

### 5. Representation-Centric Interpretation of Model Differences
Rather than treating GLM and DL as competing predictive systems, this study frames them as:

- **GLM**: hypothesis-driven statistical inference of task-evoked activation  
- **DL**: data-driven predictive representation learning model  

The comparison therefore focuses on **alignment between statistical inference maps and learned representations**, not raw predictive performance.

### 6. Controlled Evaluation Protocol for Reproducibility
All analyses are performed under a unified experimental protocol:
- Leave-One-Subject-Out (LOSO) cross-validation
- Strict subject-level separation (no data leakage)
- Permutation-based null models for significance testing
- Identical preprocessing and feature extraction pipelines

This ensures that observed differences reflect **model class behavior rather than pipeline differences**.

---

### 🔑 Summary of Contribution

In summary, this study provides a controlled methodological framework for:

- Comparing statistical inference (GLM) and predictive modeling (DL) under identical conditions  
- Characterizing model behavior under small-sample fMRI constraints  
- Quantifying overfitting and stability as primary dependent variables  
- Evaluating spatial correspondence between inference-driven and data-driven representations  

Rather than introducing new algorithms, this work characterizes the **behavioral and representational limits of existing modeling approaches under constrained neuroimaging regimes**.

---

## 🎯 Objectives

1. Perform standardized preprocessing and motion-aware quality control using FSL pipelines.  
2. Fit GLM models to obtain statistical activation maps (Incongruent > Congruent).  
3. Train regularized ML and DL models under extreme sample-size constraints.  
4. Evaluate cross-subject generalization using Leave-One-Subject-Out (LOSO).  
5. Quantify overfitting via training–validation divergence and learning curves.  
6. Quantify spatial correspondence between GLM activation maps and DL attribution maps.  
7. Perform permutation testing to establish empirical chance performance.

---

## 📊 Experimental Design

This study follows a strictly controlled evaluation protocol:

### Data Splitting Strategy
- **Leave-One-Subject-Out (LOSO) Cross-Validation**: used for all predictive modeling and evaluation.
- **Strict subject-level separation**: ensures no subject leakage between training and test sets.

### Baseline Models
- **General Linear Model (GLM)**: statistical inference baseline for task-evoked activation.
- **Logistic Regression**: linear predictive baseline using extracted features.

### Deep Learning Models
- **Shallow Multi-Layer Perceptron (MLP)** on extracted features  
- **Regularized CNNs (1D and 3D)** with dropout and weight decay constraints  

### Evaluation Protocol
- **Permutation Testing**: ≥1000 label shuffles for empirical null distributions  
- **Cross-validation variance reporting** across LOSO folds  
- **Learning curve analysis** to quantify overfitting dynamics (train vs validation divergence)

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
# Feature representations include both voxel-wise maps and reduced-dimensional embeddings to mitigate overfitting in high-dimensional space

# Cross-validation strategy
- Leave-one-subject-out (LOSO)
- Stratified k-fold within subjects
- Proper train/validation/test splits
```

### Phase 5: Deep Learning Implementation under Small-Sample Constraints
```python
# Architecture considerations (designed for small N=26 regime)
- Shallow network designs optimized to characterize baseline overfitting
- Convolutional layers (1D and 3D) for spatial and temporal features
- Explicit dropout, L2 regularization, and batch normalization to measure generalization curves
- Systematic benchmarking across varying sample subsets (degradation analysis)

# Model variants
1. Multi-layer Perceptron (MLP) on classical feature extractions
2. 1D CNN on voxel time series
3. 3D CNN on full-brain activation maps (measuring overfitting under heavy spatial constraints)
4. Ensemble methods combining classical and deep architectures
```

### Phase 6: Comparative & Spatial Attribution Evaluation
```python
# Generalization metrics
- Classification accuracy, precision, recall, F1-score
- ROC-AUC and precision-recall curves  
- Cross-validation variance and stability (using Leave-One-Subject-Out)
- Learning curves: training vs. validation trajectory analysis (quantifying generalization gap)

# Spatial overlap analysis
- Spatial overlap: Comparing GLM statistical activation maps (Z-maps) with DL-derived spatial attribution maps. Overlap is quantified using similarity metrics such as Pearson correlation or Dice coefficient.
- Methodological comparison: GLM as a statistical baseline vs. DL as a distributed feature learner
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

## 📈 Expected Outcomes

- GLM is expected to produce stable and interpretable activation maps consistent with established cognitive control networks involved in cognitive control tasks.
- Deep learning models are expected to show higher variance across cross-validation folds due to limited sample size.
- Spatial overlap between GLM and DL-derived maps will be evaluated rather than assumed.
- Model performance will be interpreted through statistical significance testing rather than fixed performance thresholds.

## 📊 Quality Metrics

### Data Quality Thresholds
```python
# Inclusion criteria (standard neuroimaging QC thresholds)
max_mean_fd = 0.5      # Maximum mean framewise displacement (mm)
max_dvars = 75         # Maximum DVARS threshold  
min_tsnr = 50          # Minimum temporal signal-to-noise ratio
max_outlier_pct = 10   # Maximum percentage outlier volumes
```

### 📊 Evaluation Strategy

Model performance is assessed using:
- Permutation-based null distributions for classification accuracy (1000+ permutations).
- Cross-subject stability (LOSO variance).
- Spatial overlap metrics between GLM and DL maps (Dice coefficient / correlation).
- Learning curve analysis to quantify overfitting behavior.

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@article{your_study_2026,
  title={A Methodological Study of GLM and Deep Learning Behavior in Small-Sample Task-Based fMRI of Cognitive Control},
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
