
# 🐶🐱 Cats vs Dogs Classification Lab (Custom CNN vs. VGG16)

## 📌 Overview
This project explores binary image classification using deep learning models to distinguish between cats and dogs. It implements a custom-built Convolutional Neural Network (CNN) and compares its performance to a transfer learning approach using the pre-trained **VGG16** model.

The workflow includes two-phase fine-tuning of VGG16, robust evaluation metrics, and thorough error analysis. It is designed to be reproducible, automated, and insightful for academic and practical ML applications.

---

## 🗂 Repository Structure

```
Cats-and-dogs-Lab-3/
├── lab3.ipynb     # 🔁 End-to-end notebook (Custom CNN + VGG16)
├── requirements.txt         # ⚙️ Required dependencies for setup
├── README.md                # 📘 Summary of project (you are here)
```

---

## 🎯 Objectives

- Build a baseline **Custom CNN** with essential layers and trainable parameters.
- Use **Transfer Learning with VGG16** to improve performance and efficiency.
- Introduce **two-phase fine-tuning** with selective layer unfreezing.
- Apply **modern deep learning best practices**: callbacks, early stopping, checkpointing.
- Evaluate both models using:
  - Accuracy, Loss
  - Confusion Matrix & Classification Report
  - ROC Curve & AUC
  - Precision-Recall Curve & Average Precision
- Visualize **failed predictions** and **model disagreement**.
- Provide clear comparison plots and statistical insights.

---

## 📁 Dataset Access

### 📦 Kaggle Dataset (Preprocessed Subset)
- Title: **Krishna CatsDogs Dataset - Lab 3**
- URL: [Kaggle Dataset](https://www.kaggle.com/datasets/kittuai/krishna-catsdogs-dataset-lab3)
- Description: 5000 manually curated and resized (150x150) images split into `train`, `validation`, and `test` sets.
- Structure:
  - `train/` – 3200 images
  - `validation/` – 800 images
  - `test/` – 1000 images (cats + 1000 dogs)

### 📥 Programmatic Access
```python
import kagglehub
kagglehub.dataset_download("kittuai/krishna-catsdogs-dataset-lab3")
```

---

## 🎓 Model Checkpoints & Histories

### 🧠 Trained VGG16 Model (HDF5 format)
- URL: [https://www.kaggle.com/datasets/kittuai/modelsh5](https://www.kaggle.com/datasets/kittuai/modelsh5)
- Includes: `best_vgg_phase2.h5`

### 📊 Training History (Pickle format)
- URL: [https://www.kaggle.com/datasets/kittuai/history](https://www.kaggle.com/datasets/kittuai/history)
- Includes: `history_vgg_phase2.pkl`

### Download via Code
```python
import kagglehub
model_path = kagglehub.dataset_download("kittuai/modelsh5")
history_path = kagglehub.dataset_download("kittuai/history")
```

---

## ⚙️ Environment Setup

### ✅ Requirements
- Python 3.9+
- TensorFlow < 2.11 (for GPU support compatibility)
- See `requirements.txt` for all dependencies.

### 🔧 Create Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 🧪 Test Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## ▶️ Running the Notebook

1. **Clone** the repository:
```bash
git clone https://github.com/kittuai/Cats-and-dogs-Lab-3-.git
cd Cats-and-dogs-Lab-3-
```

2. **Install** dependencies:
```bash
pip install -r requirements.txt
```

3. **Open** the notebook and **run all cells**:
- `lab3.ipynb` will:
  - Download the dataset and models automatically.
  - Train and evaluate models.
  - Generate visual diagnostics and comparisons.

---

## 📊 Performance Summary

| Model      | Accuracy | Loss  | AUC   | Avg Precision |
|------------|----------|-------|-------|----------------|
| Custom CNN | ~0.68    | ~0.68 | ~0.73 | ~0.69           |
| VGG16      | ~0.95    | ~0.26 | ~0.98 | ~0.96           |

> VGG16 significantly outperforms the baseline CNN in all evaluation metrics.

---

## 📈 Visual Insights
- Confusion matrices highlight clear classification improvements.
- ROC and PR curves show better true/false positive control with VGG16.
- Misclassified and disputed examples offer **interpretability** of where models fail.

---

## 📌 Why This Project Stands Out

- ✅ Two models with contrasting architectures for learning comparison
- ✅ Transfer learning executed in **phased fine-tuning**, not just feature extraction
- ✅ Use of **model checkpointing**, **early stopping**, **adaptive learning rate**
- ✅ Rich **interpretation of visual errors**
- ✅ All results, models, and data are **reproducible** and publicly accessible
- ✅ Full automation: No manual downloads or configurations needed

---

## 🔗 GitHub Repository
- 📍 URL: [https://github.com/kittuai/Cats-and-dogs-Lab-3-](https://github.com/kittuai/Cats-and-dogs-Lab-3-)

---

## 📬 Contact

For any academic queries, collaboration, or feedback:
- Raise an issue on GitHub
- Connect via your preferred academic channel

---

© Krishna Reddy | Lab 3 – Deep Learning (Cats vs Dogs)
