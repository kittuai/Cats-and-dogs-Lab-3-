# 🐶 Cats vs Dogs – Deep Learning Binary Classification (Lab 3)

## 🔍 Overview

This project presents a professional, end-to-end solution for binary image classification using deep learning, focused on distinguishing between images of cats and dogs. It compares two architectures:

- A custom Convolutional Neural Network (CNN) designed from scratch.
- A fine-tuned version of **VGG16**, a popular pre-trained model from the ImageNet challenge.

The project follows modern best practices in training, evaluation, model saving, and visualization.

---

## 📁 Project Structure

```
Cats-and-dogs-Lab-3/
│
├── Lab3_Cats_vs_Dogs.ipynb      # Jupyter Notebook with full workflow
├── requirements.txt             # All required packages
├── README.md                    # This file
└── models/                      # Folder for best model weights (.keras / .h5)
```

---

## 🧠 Models Used

| Model       | Accuracy | Loss   |
|-------------|----------|--------|
| Custom CNN  | 68.15%   | 0.6855 |
| VGG16       | **94.60%** | 0.2648 |

- Evaluation includes Accuracy, Loss, Confusion Matrix, ROC-AUC, PR Curve, and Misclassified Visualization.
- Smart callbacks: EarlyStopping, ReduceLROnPlateau, and BestModelSaver were used.

---

## 📦 Dataset Access (via Kaggle)

**Dataset Name:** `kittuai/krishna-catsdogs-dataset-lab3`  
**Kaggle Dataset URL:** https://www.kaggle.com/datasets/kittuai/krishna-catsdogs-dataset-lab3

### 🔽 Steps to Download Automatically

The code uses `kagglehub` for automatic download and caching.

1. Make sure you have the `kagglehub` package:
    ```bash
    pip install kagglehub[pandas-datasets]
    ```

2. Add your Kaggle API credentials:
    ```bash
    kagglehub login --username YOUR_USERNAME --key YOUR_API_KEY
    ```

3. Inside the notebook, datasets and models will auto-download using:
    ```python
    kagglehub.dataset_download("kittuai/krishna-catsdogs-dataset-lab3")
    ```

---

## 🧰 Environment Setup

> Recommended: Python 3.10+, TensorFlow 2.10 (GPU compatible)

### ✅ Create and activate a new environment:

```bash
conda create -n dogs-cats-env python=3.10 -y
conda activate dogs-cats-env
```

### 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Notebook

1. Clone the repo:

```bash
git clone https://github.com/kittuai/Cats-and-dogs-Lab-3-.git
cd Cats-and-dogs-Lab-3-
```

2. Start Jupyter:

```bash
jupyter notebook Lab3_Cats_vs_Dogs.ipynb
```

3. Follow the notebook sequentially. All models will download or train as needed.

---

## 💾 Trained Models on Kaggle

**Checkpoint Dataset:** `kittuai/modelsh5`  
**Includes:** `best_model_final.keras`, `best_vgg_phase2.h5`

Access via:
```python
kagglehub.dataset_download("kittuai/modelsh5")
```

The models will be copied to the `models/` directory automatically.

---

## 📊 Results Summary

| Metric            | Custom CNN | VGG16         |
|-------------------|------------|---------------|
| Accuracy          | 68.15%     | **94.60%**    |
| ROC-AUC           | 0.73       | **0.99**      |
| Precision-Recall  | 0.74       | **0.95**      |
| Classification F1| 0.67       | **0.95**      |

---

## 🧪 Features and Best Practices

- **Training Pipeline:** Split, augmentation, early stopping, fine-tuning
- **Evaluation:** ROC, PR, Confusion Matrix, Misclassifications
- **Reproducibility:** Model saving, history saving (`.pkl`)
- **Visualization:** Matplotlib, seaborn, tabulate, PR/ROC curves

---

## 🧠 Author

**Krishna Reddy**  
GitHub: [kittuai](https://github.com/kittuai)  
Kaggle: [@kittuai](https://www.kaggle.com/kittuai)

---

## 📌 Final Notes

- The entire pipeline runs with no manual intervention once the environment is set.
- All datasets and models are public and auto-downloadable.
- Every section is reproducible and modular.
- Evaluations are thorough and visualized professionally.

> This project sets a benchmark for clean, explainable, and automated binary image classification.
