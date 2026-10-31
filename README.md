# Lead-Aware Explainable Deep Learning for Multi-Label Arrhythmia Detection on 12-Lead ECG Signals

## Overview

This project presents a deep learning-based approach for detecting multiple cardiac conditions from 12-lead ECG (Electrocardiogram) signals.

The system uses a hybrid **CNN-BiLSTM architecture with Channel Attention** to analyse ECG recordings and classify five diagnostic categories simultaneously. In addition to predictions, the project incorporates Explainable AI (XAI) techniques to understand which ECG leads and signal regions influence the model's decisions.

The primary focus is not only on classification performance but also on evaluating the reliability of model explanations.

## Problem Statement

Many existing automated ECG classification systems have limitations such as:

- Reliance on single-lead ECG recordings.
- Prediction of only one diagnostic category per recording.
- Limited interpretability of deep learning predictions.
- Lack of validation of lead-level explanations.

This project addresses these limitations by analysing all 12 ECG leads jointly, supporting multi-label classification and comparing multiple explainability techniques.

## Diagnostic Categories

The model predicts five diagnostic superclasses:

| Class | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Changes |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

Since multiple conditions can coexist in a single recording, the problem is formulated as a multi-label classification task.

## Dataset

### Primary Dataset: PTB-XL

The primary model uses 12-lead ECG recordings with the following characteristics:

- Total recordings: 25,770
- Number of leads: 12
- Recording duration: 10 seconds
- Sampling frequency: 500 Hz
- Samples per lead: 5,000

Diagnostic codes are mapped into five superclasses to generate multi-hot label vectors.

### Baseline Dataset: MIT-BIH + INCART

A separate combined heartbeat dataset is used to develop a baseline feed-forward neural network.

- Total heartbeat samples: 712,822
- Features per heartbeat: 32
- Classification categories: 5

The baseline is used to study classification performance and the effect of class weighting.

## Methodology

### 1. Data Preprocessing

- Mapping AHA diagnostic codes to five superclasses.
- Generating multi-hot label vectors.
- Extracting ECG signals from HDF5 files.
- Batch-wise extraction with checkpoints.
- Per-lead Z-score normalisation.
- Memory-mapped arrays for memory-efficient processing.
- Stratified 80:20 train-test split.

### 2. Baseline Model

A feed-forward neural network is implemented using PyTorch.

- Two hidden layers.
- ReLU activation.
- Batch Normalisation.
- Dropout regularisation.
- Class weighting to handle imbalanced heartbeat categories.

### 3. Primary Deep Learning Model

The main architecture combines three components:

**1D CNN**

Extracts waveform features independently from each ECG lead using shared convolutional weights.

**Channel Attention**

Learns weights for the 12 leads based on their extracted features.

**Bidirectional LSTM**

Captures sequential relationships between lead-level representations by processing information in both forward and backward directions.

The final classification layer produces five independent outputs using sigmoid probabilities.

Weighted Binary Cross-Entropy with Logits Loss is used for multi-label classification.

### 4. Explainable AI Techniques

Three techniques are implemented and compared:

**Channel Attention**

Examines the learned importance weights assigned to individual ECG leads.

**Grad-CAM**

Uses gradients to generate heatmaps highlighting the signal regions that contribute to a particular prediction.

**Occlusion Analysis**

Removes one ECG lead at a time and measures the decrease in predicted probability. A larger decrease indicates greater importance of that lead for the prediction.

### 5. Threshold Tuning and Evaluation

Separate decision thresholds are tuned for each diagnostic category to improve classification performance.

Evaluation includes:

- Macro F1-score
- Per-class F1-score
- AUROC
- Per-label accuracy
- Exact-match accuracy

## Results

### Primary Model

| Metric | Result |
|---|---:|
| Tuned Macro F1-score | 0.623 |
| AUROC range | 0.851–0.953 |
| Number of diagnostic classes | 5 |
| Number of ECG leads | 12 |

### Baseline Model

| Metric | Result |
|---|---:|
| Accuracy | 97.37% |
| Macro F1-score | 0.708 |

Accuracy alone can be misleading when dealing with highly imbalanced medical datasets, which is why macro F1-score and per-class evaluation are also considered.

## Key Findings

- Channel-attention weights were almost uniform and did not provide reliable class-specific lead importance.
- Grad-CAM generated useful explanations for individual recordings, although aggregated maps showed a bias towards lead V6.
- Occlusion analysis produced more distinct lead-importance patterns.
- Occlusion patterns were also examined across different model architectures for consistency.
- Increasing model depth did not necessarily improve classification performance or explanation quality.

These findings highlight the importance of validating explainability methods rather than relying on visual explanations alone.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning and model training |
| NumPy | Numerical operations and memory-mapped arrays |
| Pandas | Dataset processing |
| h5py | HDF5 ECG data extraction |
| Scikit-learn | Data splitting and evaluation metrics |
| Matplotlib | ECG visualisation and explainability plots |
| Google Colab | Development and GPU execution |
| Google Drive | Dataset and model storage |

## Project Structure

```text
ECG-Arrhythmia/
│
├── Dataset/
│   ├── records/
│   ├── metadata.csv
│   ├── code.csv
│   └── kaggle dataset/
│
├── processed_data/
│   ├── checkpoints/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── X_kaggle.npy
│   └── y_kaggle.npy
│
├── waveform_images/
│
├── best_cnn_final_v2.pt
├── baseline_model.pt
│
└── ecg_data_pipeline.ipynb
```

*Note: The dataset and generated model files are stored separately and are not necessarily included in the GitHub repository.*

## Installation and Execution

The project is developed using Google Colab.

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <repository-folder>
```

**2. Install dependencies**

```bash
pip install numpy pandas h5py tqdm
pip install torch scikit-learn matplotlib
```

**3. Upload the notebook to Google Colab**

Open `ecg_data_pipeline.ipynb` in Google Colab.

**4. Configure Google Drive**

Mount Google Drive and update the dataset paths in Cell 1 according to the folder structure.

**5. Prepare the datasets**

Ensure the required PTB-XL HDF5 recordings, metadata, diagnostic code mapping and baseline dataset CSV files are available in their respective directories.

**6. Execute the notebook**

Run the notebook cells sequentially to perform preprocessing, train the baseline and primary models, evaluate performance and generate explainability visualisations.

The trained model checkpoints and processed arrays must be available when running the evaluation and demonstration sections.

## Limitations

- The system is trained and evaluated on public datasets.
- Performance on unseen patient populations has not been established.
- The project uses a single stratified train-test split.
- The model is an offline research prototype and has not been clinically deployed.
- Predictions should not be treated as a substitute for professional medical diagnosis.

## Future Scope

- Validation using independent clinical ECG datasets.
- Improved lead-attention mechanisms.
- Exploration of transformer-based ECG architectures.
- Real-time ECG signal analysis.
- Further investigation of explanation reliability and clinical interpretability.

## Author

**Archi Garg**

Registration Number: 23BDS0155

B.Tech – Data Science

## Acknowledgement

This project was developed as part of Project-I under the guidance of Dr. Bhawana Tyagi at Vellore Institute of Technology (VIT).

---

*This project is intended for academic and research purposes only.*
