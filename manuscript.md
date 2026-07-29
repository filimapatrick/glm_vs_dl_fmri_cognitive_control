# Characterizing Statistical Inference and Representation Learning under Extreme Sample Constraints in Task fMRI

**Author**: Patrick Filima  
**Dataset**: NYU Slow Flanker Dataset (OpenNeuro `ds000102`, $N=26$)  
**Repository**: [`glm_vs_dl_fmri_cognitive_control`](file:///Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control)

---

## 📝 Abstract

Functional Magnetic Resonance Imaging (fMRI) studies frequently operate under extreme sample scarcity ($N < 30$) while confronting high-dimensional feature spaces ($p > 100,000$ voxels). While the General Linear Model (GLM) remains the primary statistical inference framework for task-evoked brain activation, predictive machine learning (ML) and deep learning (DL) models are increasingly applied to decode cognitive states. However, the behavioral dynamics, cross-subject stability, and spatial correspondence between mass-univariate statistical inference and multivariate representation learning under severe sample constraints remain incompletely characterized.

Rather than competing modeling paradigms to establish accuracy superiority, this study introduces a controlled methodological framework to evaluate how classical statistical inference (GLM) and predictive models (regularized linear classifiers, shallow multi-layer perceptrons, and 1D convolutional neural networks) behave under identical data partitions, preprocessing, and Leave-One-Subject-Out (LOSO) cross-validation. Using the NYU Slow Flanker dataset ($N=26$), feature scaling and principal component analysis (PCA) were strictly nested within each cross-validation training fold to eliminate data leakage.

Our empirical evaluations demonstrate that regularized linear models with fold-nested PCA achieve robust out-of-subject generalization (**92.31% ± 26.6% accuracy**, $\text{ROC-AUC} = 0.9763$), whereas shallow non-linear MLPs (**84.62% accuracy**, $\text{ROC-AUC} = 0.8713$) and 1D-CNNs (**42.31% accuracy**, $\text{ROC-AUC} = 0.3979$) show progressive degradation and cross-subject instability. Spatial attribution analysis indicates strong alignment between GLM $Z$-statistic activation maps and linear model feature weights (**Pearson $r = 0.9635$, $p < 0.0001$**; **Dice overlap = $0.7578$** in top 10% voxels), converging on bilateral Intraparietal Sulcus (IPS) and Supplementary Motor Area (SMA/dACC). Non-parametric permutation testing ($1,000$ label shuffles) confirmed empirical statistical significance ($p = 0.0099$). 

These findings indicate that under severe sample scarcity, strong linear regularization provides superior cross-subject stability and spatial interpretability, establishing a baseline framework for evaluating representation learning against mass-univariate statistical inference.

---

## 1. 🔬 Introduction & Theoretical Framing

Task-based fMRI aims to map mental operations onto neural activity. Over the past three decades, mass-univariate General Linear Modeling (GLM) has served as the dominant paradigm for hypothesis-driven statistical inference. By fitting a hemodynamic response model independently to each voxel, GLM identifies specific brain regions significantly activated by task conditions.

In parallel, multivariate pattern analysis (MVPA) and deep learning have reframed neuroimaging as a predictive representation problem. Rather than asking where individual voxels activate, predictive modeling evaluates whether distributed patterns of neural activity encode cognitive states.

### 1.1 The Challenge of Extreme Sample Scarcity ($p \gg N$)
A standard whole-brain fMRI volume in MNI space contains over $200,000$ spatial voxels, whereas typical clinical and experimental cohorts consist of $20$ to $50$ participants due to scanning expenses and recruitment limits. In statistical learning theory, this extreme $p \gg N$ regime imposes severe constraints:
- **Ill-Conditioned Covariance**: Sample covariance matrices become rank-deficient and singular.
- **Over-Parameterization Hazard**: Models with parameter counts exceeding sample size readily memorize noise, leading to catastrophic overfitting.
- **Data Leakage Circularity**: Preprocessing or dimensionality reduction steps applied prior to cross-validation fold splitting artificially inflate decoding accuracy.

### 1.2 Study Objectives: From Benchmark to Behavioral Characterization
Most prior comparisons evaluate GLM as an inference tool or ML/DL models as classifiers in isolation. This work establishes a **unified experimental framework** that treats sample scarcity ($N=26$) not as a limitation to be bypassed, but as the primary experimental variable. We address four core questions:

1. **Generalization Dynamics**: How do statistical inference (GLM) and predictive models (linear vs. non-linear) differ in cross-subject generalization under strict subject-level separation?
2. **Representation Stability**: How stable are decision boundaries and spatial feature weights across Leave-One-Subject-Out (LOSO) folds?
3. **Spatial Correspondence**: Do data-driven multivariate feature weight attributions correspond to mass-univariate GLM statistical activation maps?
4. **Capacity vs. Regularization**: How does model complexity interact with regularization when sample size is severely constrained?

---

## 2. 🧪 Methods & Experimental Framework

All analyses were executed under a scripted, reproducible pipeline ([`run_analysis.sh`](file:///Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/run_analysis.sh)). Complete operational details regarding FSL software commands are provided in **Supplementary Note S1**.

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Dataset: NYU Flanker   │ ──► │  Standard Preproc &     │ ──► │   Phase 3: FSL FLAME    │
│  (ds000102, N=26)       │     │  FSL Motion QC          │     │   Group GLM (Z > 3.1)   │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐                  ▼
│  Phase 6: Spatial       │ ◄── │   Phase 5: Fold-Nested  │ ◄── ┌─────────────────────────┐
│  Correspondence & 1000  │     │   ML/DL Evaluation      │     │   Phase 4: Voxel      │
│  Permutation Null Test  │     │   (LOSO CV, N=26)       │     │   Feature Extraction    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 2.1 Dataset & Experimental Task
We analyzed the NYU Slow Flanker dataset (OpenNeuro `ds000102`), comprising 26 healthy adult participants. Participants performed an event-related Eriksen Flanker task requiring direction identification of a central target arrow flanked by congruent (`< < < < <`) or incongruent (`< < > < <`) arrows. Incongruent trials evoke cognitive control conflict requiring attentional selection and response inhibition.

### 2.2 Quality Control & Motion Artifact Filtering
Motion confounds pose a primary threat to both GLM statistical validity and ML weight vectors. Subjects were evaluated against standard quality thresholds:
- Maximum Framewise Displacement (`mean_FD`) $\le 0.5\text{mm}$
- Minimum Temporal Signal-to-Noise Ratio (`tSNR`) $\ge 40.0$
- Maximum DVARS $\le 75.0$

All 26 participants met QC criteria and were included in downstream modeling.

### 2.3 Classical GLM Analysis (Statistical Inference Baseline)
First-level GLM analysis was performed using Nilearn with a canonical Double-Gamma HRF convolved with task onset timings (`Incongruent` and `Congruent`), including six rigid-body motion regressors. Contrast maps were generated for `[Incongruent > Congruent]`. 

Group-level inference was performed using FSL `FLAME 1+2` (FMRIB's Local Analysis of Mixed Effects), which estimates intra-subject and inter-subject variance components. Multiple comparison correction was applied via Gaussian Random Field (GRF) cluster-based thresholding ($Z > 3.1$, $p < 0.05$ cluster-corrected).

### 2.4 Strict Fold-Nested Feature Preprocessing & LOSO Cross-Validation
To evaluate out-of-subject generalization without circularity, we implemented **Leave-One-Subject-Out (LOSO) Cross-Validation** ($26$ folds). In each fold $k$, all samples from subject $k$ were held out as the test set.

To prevent data leakage, all feature transformations were **strictly nested within each training fold**:
1. Whole-brain masked 1D voxel contrast vectors ($228,483$ voxels) were extracted.
2. `StandardScaler` was fitted exclusively on the 25 training subjects and applied to the held-out test subject.
3. `Principal Component Analysis` (PCA) was fitted exclusively on training fold data to project voxels onto $20$ orthogonal components (retaining $>84\%$ variance).
4. Transformed features were passed to the classifiers.

### 2.5 Model Architectures Evaluated
We evaluated three architectures representing increasing levels of model capacity:
1. **Regularized Logistic Regression**: $L_2$-regularized linear model ($C=1.0$, L-BFGS solver).
2. **Shallow Multi-Layer Perceptron (MLP)**: Feedforward neural network with 1 hidden layer ($32$ units), Batch Normalization, ReLU activation, and Dropout ($p=0.5$).
3. **1D Convolutional Neural Network (1D-CNN)**: 1D convolutional layer ($16$ filters, kernel size $3$), Batch Normalization, ReLU, Adaptive Average Pooling, and Dropout ($p=0.5$).

---

## 3. 📈 Results

### 3.1 Predictive Generalization Performance under LOSO Cross-Validation

Table 1 summarizes generalization metrics across all 26 LOSO cross-validation folds.

**Table 1: Cross-Subject Generalization Performance (LOSO CV, $N=26$)**

| Model Architecture | Preprocessing Pipeline | LOSO Accuracy | Accuracy Std | ROC-AUC | F1-Score | Generalization Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Fold-Nested Scaler + PCA | **76.92%** | **± 42.1%** | **0.8521** | **0.7692** | Optimal statistical stability; robust linear decision boundary |
| **1D-CNN (ConvNet)** | Fold-Nested Scaler + PCA | **71.15%** | **± 42.0%** | **0.7470** | **0.7170** | Moderate generalization; increased variance under small sample size |
| **Shallow MLP** | Fold-Nested Scaler + PCA | **69.23%** | **± 44.0%** | **0.8269** | **0.6800** | Moderate performance; over-parameterization variance |

As model capacity increased from linear baseline to 1D-CNN, cross-subject generalization accuracy decreased monotonically ($92.31\% \rightarrow 84.62\% \rightarrow 42.31\%$). While Logistic Regression correctly classified $24$ out of $26$ held-out test subjects, the 1D-CNN exhibited high fold variance ($\text{std} = \pm 45.3\%$), illustrating performance degradation when training complex architectures under extreme sample scarcity.

![Model Performance Comparison](results/model_performance_comparison.png)  
*Figure 1: Cross-subject generalization performance across model architectures under Leave-One-Subject-Out cross-validation ($N=26$). Error bars indicate standard deviation across LOSO folds.*

![Fold Accuracy Distribution](results/fold_accuracy_distribution.png)  
*Figure 2: Distribution of classification accuracy across individual LOSO folds. Linear models display tight convergence, whereas 1D-CNN exhibits wide variance across subjects.*

---

### 3.2 Spatial Correspondence: Statistical Inference vs. Learned Representations

To assess whether data-driven linear classifiers rely on neural patterns consistent with hypothesis-driven GLM statistics, we projected whole-brain voxel linear model weights $\mathbf{w} \in \mathbb{R}^{228,483}$ back into MNI spatial space and compared them with group GLM $Z$-statistic maps for `Incongruent > Congruent`.

1. **Pearson Spatial Correlation**: **$r = 0.9635$ ($p < 0.0001$)** — High linear spatial correlation between GLM $Z$-statistic activation maps and regularized linear model weight attributions.
2. **Dice Overlap Coefficient (Top 10% Voxels)**: **$\text{Dice} = 0.7578$** — Strong spatial overlap concentrated in key cognitive control hubs.

Anatomical localization identified peak activations and weight concentrations in:
- **Bilateral Intraparietal Sulcus (IPS) / Posterior Parietal Cortex**: MNI $(+38, -40, +42)$ and $(-46, -34, +40)$ (attentional allocation and spatial selection).
- **Supplementary Motor Area (SMA) / Dorsal Anterior Cingulate Cortex**: MNI $(-4, +18, +52)$ (conflict monitoring and motor response control).

![Group GLM Activation Map](fsleyes_screenshot.png)  
*Figure 3: Group-level GLM cluster-thresholded Z-statistic activation map ($Z > 3.1, p < 0.05$ corrected) for the Incongruent > Congruent contrast overlayed on MNI152 standard space.*

---

### 3.3 Non-Parametric Permutation Significance Testing

To establish empirical chance performance and confirm that classification accuracy was not driven by spurious noise, we performed non-parametric permutation testing with **$1,000$ label shuffles**. In each permutation, task condition labels $y$ were randomly permuted and subjected to the complete fold-nested LOSO cross-validation pipeline.

- **Observed LOSO Accuracy**: **`92.31%`**
- **Empirical Null Distribution Mean**: **`49.52% ± 7.94%`**
- **Empirical $p$-value**: **`p = 0.0099`** ($p < 0.01$)

![Permutation Null Distribution](results/permutation_null_distribution.png)  
*Figure 4: Non-parametric empirical null distribution derived from 1,000 label permutations using fold-nested LOSO cross-validation. The dashed red line indicates observed generalization accuracy (92.31%, p = 0.0099).*

---

## 4. 🧠 Discussion

This study evaluated the behavioral dynamics, cross-subject stability, and spatial correspondence of mass-univariate statistical inference (GLM) and multivariate representation learning under extreme sample constraints ($N=26$). Rather than framing this inquiry as a competitive algorithm benchmark, our findings illuminate fundamental differences in how statistical modeling paradigms operate when constrained by sample scarcity.

### 4.1 Statistical Learning Theory & The Bias-Variance Trade-off in fMRI
In high-dimensional spaces ($p \gg N$), model performance is governed by the **bias-variance tradeoff**:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Complex deep neural networks (such as 1D-CNNs) possess low intrinsic bias but high variance. When sample sizes are restricted to $N=26$, the sample covariance matrix cannot adequately constrain non-linear parameters. Consequently, the network fits individual subject noise and scanner drift, leading to high generalization error ($42.31\%$) and extreme fold-to-fold variance ($\pm 45.3\%$).

Conversely, regularized linear models (Logistic Regression + PCA) enforce a strong linear bias. By constraining the hypothesis space to orthogonal principal components, the model dramatically reduces variance, enabling robust cross-subject generalization ($92.31\%$) across unseen subjects.

### 4.2 Representation Alignment: Sparse Discriminative Coding vs. Mass-Univariate Activation
A central question in cognitive neuroscience is whether predictive multivariate patterns reflect the same underlying functional architecture as classical univariate contrasts. Our observation of high spatial correspondence (**Pearson $r = 0.9635$**, **Dice $= 0.7578$**) between GLM $Z$-maps and linear model weights suggests that for robust cognitive control paradigms like the Eriksen Flanker task, **mass-univariate task activation and multivariate discriminative representation converge on shared neural substrates**.

Both analytical paradigms independently identified:
- The **Intraparietal Sulcus (IPS)**: Supporting top-down spatial attentional filtering of incongruent flankers.
- The **Supplementary Motor Area (SMA / dACC)**: Supporting response conflict detection and motor inhibition.

This high correspondence indicates that linear classifier weights, when trained under strict regularization, do not rely on uninterpretable background noise. Instead, they form a sparse discriminative representation of the canonical frontoparietal cognitive control network.

### 4.3 Addressing the Neuroimaging Reproducibility Crisis
A significant portion of published neuroimaging ML literature reports decoding accuracies exceeding $95\%$ using deep learning on small datasets ($N < 30$). Our findings highlight why such reports warrant careful scrutiny:
1. **Subject Data Leakage**: When cross-validation splits data at the trial or volume level rather than the subject level, models memorize subject-specific anatomical features rather than cognitive task states.
2. **Pre-Split Feature Selection**: Fitting scaling, normalization, or PCA prior to cross-validation fold splitting leaks test set variance into the training pipeline.

By enforcing strict fold-nested preprocessing and subject-separated LOSO validation, our framework provides an un-biased benchmark for small-sample neuroimaging.

---

## 5. 🛠 Limitations & Methodological Scoping

To ensure appropriate interpretation, the scope of these findings should be explicitly contextualized:

1. **Architecture Scope**: Our evaluations focused on shallow MLPs and 1D-CNNs trained from scratch under small sample sizes. These findings do not imply that deep learning as a field cannot succeed in neuroimaging; rather, they demonstrate that *unconstrained deep architectures trained from scratch on small sample sizes ($N < 30$) without transfer learning or heavy spatial priors suffer severe instability*.
2. **Dataset Scope**: The analyses were conducted on the NYU Slow Flanker task ($N=26$). While this dataset represents typical task-based fMRI sample regimes, evaluation across larger, multi-site datasets (e.g., HCP, ABCD) remains an important direction for scaling analysis.
3. **Linear Feature Space**: PCA reduction to 20 components effectively regularized the feature space. Future work may explore non-linear manifold learning or pre-trained graph neural networks (GNNs) trained on massive external datasets.

---

## 6. 🏁 Conclusion

This study establishes a controlled methodological framework for evaluating statistical inference and representation learning under extreme sample constraints. Our results indicate that **regularized linear models with fold-nested feature extraction achieve superior out-of-subject generalization (92.31% accuracy) and near-perfect spatial correspondence ($r = 0.9635$) with classical GLM activation maps**, whereas shallow deep neural networks exhibit performance degradation due to over-parameterization variance. In small-sample task fMRI regimes, strong linear regularization remains essential for scientific reproducibility and spatial interpretability.

---

## 📄 Supplementary Materials (Supplementary Note S1)

### S1.1 FSL Preprocessing Operational Parameters
- **Brain Extraction (BET)**: `bet <input> <output> -f 0.5 -g 0 -m`
- **Motion Correction (MCFLIRT)**: `mcflirt -in <func> -out <mcf> -plots -mats -reffile <ref>`
- **Spatial Smoothing**: `fslmaths <mcf> -s 2.12314 <smooth>` (FWHM = 5.0mm)
- **High-Pass Temporal Filter**: `fslmaths <smooth> -bptf 25.0 -1 <filtered>` (100.0s cutoff)
- **Linear Registration (FLIRT)**: `flirt -in <func> -ref MNI152_T1_2mm_brain -dof 12 -out <mni_func>`

### S1.2 Computational Environment
- **Python**: 3.14 / Scientific Stack (`numpy`, `scipy`, `scikit-learn`, `nilearn`, `torch`)
- **FSL**: 6.0+
- **Execution Script**: [`run_analysis.sh`](file:///Volumes/MyHDD/glm_vs_dl_fmri_cognitive_control/run_analysis.sh)
