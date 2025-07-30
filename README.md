
# Cats vs Dogs – Advanced Image Classification with CNN and VGG16

This repository contains an in-depth analysis and implementation of binary image classification using a custom Convolutional Neural Network (CNN) and a fine-tuned VGG16 model, trained on a balanced subset of 5000 images (cats and dogs) from the original Kaggle dataset.

## 📁 Repository Structure

```
Cats-and-dogs-Lab-3/
│
├── Lab3_CNN_VGG16.ipynb         # Jupyter notebook with full pipeline
├── requirements.txt             # All required packages and versions
├── README.md                    # Project overview and detailed usage
└── models/                      # Directory to store trained model weights
```

---

## 🧠 Project Overview

This project aims to compare the performance of:

- A custom-built Convolutional Neural Network (CNN)
- A fine-tuned transfer learning model using VGG16

Each model is trained, validated, and tested using carefully preprocessed image data. Evaluation is conducted with metrics such as Accuracy, Loss, Precision, Recall, F1-Score, AUC, Confusion Matrix, PR & ROC Curves, and visual misclassification analysis.

---

## 📊 Dataset Details

- **Source:** Kaggle's Dogs vs Cats dataset
- **Curated Subset Size:** 5000 images (train, val, test)
- **Download Path (via KaggleHub):** `kittuai/krishna-catsdogs-dataset-lab3`
- **Structure:**
    ```
    dataset/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── validation/
    └── test/
    ```

The notebook includes an auto-downloading script to fetch the dataset using `kagglehub`.

---

## 📦 Model Checkpoints and Artifacts

### 🔹 Pretrained Models (VGG16 Checkpoints)
- Phase 1: `best_vgg_phase1.h5`
- Phase 2 (fine-tuned): `best_vgg_phase2.h5`

### 🔹 Download from Kaggle
All models and histories are available in the public Kaggle dataset:
- URL: https://www.kaggle.com/datasets/kittuai/modelsh5
- Programmatic Access:
```python
import kagglehub
kagglehub.dataset_download("kittuai/modelsh5")
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/kittuai/Cats-and-dogs-Lab-3-.git
cd Cats-and-dogs-Lab-3-
```

### 2. Create and Activate Environment
```bash
conda create -n catsdogs python=3.9 -y
conda activate catsdogs
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

> ⚠️ If using GPU, ensure TensorFlow < 2.11 is used and the compatible CUDA/cuDNN versions are installed.

---

## 🚀 How to Run the Notebook

Launch the notebook and run all cells sequentially.

```bash
jupyter notebook Lab3_CNN_VGG16.ipynb
```

The notebook will:
- Auto-download the dataset and models if not found locally
- Train the models (or load saved ones)
- Generate evaluations, visualizations, and comparison tables

---

## 📈 Evaluation Metrics

Each model is evaluated using:
- Accuracy and Loss (Test Set)
- Confusion Matrix (Visual)
- Precision, Recall, F1-score
- AUC (ROC Curve)
- Average Precision (PR Curve)
- Misclassified Image Visualization
- Comparison of Custom CNN vs. VGG16
- Correct/Wrong prediction overlap analysis

---

## 📌 Key Highlights

- **Auto Resumable Pipelines:** Automatically downloads missing datasets and models
- **Model Checkpointing:** Uses `ModelCheckpoint`, `EarlyStopping`, and custom `BestModelSaver`
- **Transfer Learning Best Practices:** Phase-wise unfreezing with learning rate scheduling
- **Visual Debugging:** All failed predictions shown with ground truth and model label
- **Professional Code Structuring:** Markdown cells, plots, interpretations throughout

---

## 📊 Performance Summary

| Model      | Accuracy | Loss   |
|------------|----------|--------|
| Custom CNN | 68.15%   | 0.6855 |
| VGG16      | 94.60%   | 0.2648 |

- VGG16 significantly outperforms the custom CNN in all evaluation metrics.
- ROC AUC and PR AUC both exceed 0.94, indicating strong binary discrimination.
- Misclassifications by the custom CNN were correctly predicted by VGG16 in 80%+ of cases.

---

## 🔍 Misclassification Insights

- The majority of errors by the custom CNN occurred due to overfitting and poor generalization.
- VGG16, after selective unfreezing of block5 layers, correctly handled complex edge cases.
- The visual heatmap and prediction overlays offer clear diagnostic value.

---

## 📚 References

- [Kaggle Dataset - Cats and Dogs](https://www.kaggle.com/datasets/kittuai/krishna-catsdogs-dataset-lab3)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)

---

## 🧪 Optional: Reproduce on Google Colab

You can open the notebook in Colab and it will download the dataset and models automatically:
```python
!pip install kagglehub
import kagglehub
kagglehub.dataset_download("kittuai/modelsh5")
```

---

## 📝 Final Notes

This notebook and repository are designed with academic submission in mind:
- Clear logic and outputs
- Automatic fallback handling
- Superior visuals and plots
- Clean and professional layout

---


