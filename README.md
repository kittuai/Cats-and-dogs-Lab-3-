# Cats vs Dogs Classification Lab (Custom CNN vs. VGG16)

## 🔍 Overview
This repository presents a full deep learning pipeline for binary image classification (Cats vs Dogs) using two approaches:
1. **Custom Convolutional Neural Network (CNN)**
2. **Transfer Learning with Pre-trained VGG16 (fine-tuned in two phases)**

The project evaluates and compares model performance using advanced metrics, visual diagnostics, and error analysis.

---

## 📁 Repository Structure

```bash
Cats-and-dogs-Lab-3/
├── lab3_cats_dogs.ipynb   # Full end-to-end Jupyter Notebook (Custom CNN + VGG16)
├── requirements.txt       # Python packages required to run the notebook
├── README.md              # This README file
```

---

## 🧠 Project Objectives

- Build and train a **Custom CNN** for image classification.
- Fine-tune a **VGG16 Transfer Learning model** in two phases.
- Use best practices: callbacks, regularization, early stopping, history tracking.
- Evaluate models using **confusion matrix**, **classification report**, **ROC & PR curves**.
- Analyze **misclassified examples** and compare model behavior.
- Demonstrate **differential performance** with in-depth insights.

---

## 📦 Dataset Access

The dataset used is a **5000-image subset** from Kaggle's Cats vs Dogs dataset.

### 🔗 Kaggle Dataset
- Dataset URL: [https://www.kaggle.com/datasets/kittuai/krishna-catsdogs-dataset-lab3](https://www.kaggle.com/datasets/kittuai/krishna-catsdogs-dataset-lab3)
- Contains: Pre-processed and resized images (150x150) in `train`, `val`, and `test` folders.

### 📥 How to Download via Code
Use the `kagglehub` library (auto-handled in the notebook):
```python
import kagglehub
kagglehub.dataset_download("kittuai/krishna-catsdogs-dataset-lab3")
```
This will automatically place the dataset in your `.cache` directory and the notebook will load it from there.

---

## 🤖 Model Checkpoint Access (Trained Weights)

### 🔗 Kaggle Model Files
- VGG16 Phase 2 Best Model (.h5): [https://www.kaggle.com/datasets/kittuai/modelsh5](https://www.kaggle.com/datasets/kittuai/modelsh5)
- VGG Training History (.pkl): [https://www.kaggle.com/datasets/kittuai/history](https://www.kaggle.com/datasets/kittuai/history)

These can be automatically downloaded in the notebook using `kagglehub`, so no manual downloading is required.

---

## 💻 Environment Setup

### Python Environment
Use the provided `requirements.txt` to set up your environment.
```bash
pip install -r requirements.txt
```

### Environment Notes
- Python 3.9+ recommended
- TensorFlow < 2.11 is required to ensure GPU compatibility
- Tested on Google Colab and local GPU (Anaconda, Windows 11)

---

## 🚀 Running the Notebook

1. Clone this repository or open directly in Colab.
2. Install dependencies using `requirements.txt`.
3. Run the notebook `lab3_cats_dogs.ipynb` from top to bottom.
4. The notebook will automatically download the dataset and model weights.

---

## 📊 Highlights

- Comparison of **Custom CNN vs. VGG16** using metrics like accuracy, loss, ROC-AUC, and average precision.
- Clear visualizations of misclassified examples.
- Smart checkpoints and early stopping for efficient training.
- Robust and reproducible evaluation with visual insights.

---

## 🔗 GitHub Repo

Project GitHub: [https://github.com/kittuai/Cats-and-dogs-Lab-3-](https://github.com/kittuai/Cats-and-dogs-Lab-3-)

---

## 📞 Contact

For any questions or feedback, feel free to reach out via GitHub Issues or Discussions.
