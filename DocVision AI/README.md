# DocVision AI 👁️📄

> **Advanced Document Intelligence & Processing Pipeline**
> 
> DocVision AI is a structured repository for processing, analyzing, and extraction of intelligence from unstructured documents (specifically **Invoices** and **Resumes**) using advanced Computer Vision, OCR, and Deep Learning techniques.

---

## 📁 Repository Structure

This repository follows a clean, modular layout optimized for machine learning research, experimentation, and production workflows:

```text
DocVision-AI/
│
├── data/
│   ├── raw/                  # Original, immutable data dumps
│   │   ├── Resume/           # Raw Resume documents (images, PDFs)
│   │   └── Invoice/          # Raw Invoice documents (images, PDFs)
│   ├── processed/            # Canonical data sets for modeling
│   └── splits/               # Train, validation, and test partition metadata
│
├── notebooks/                # Jupyter notebooks for EDA and quick prototyping
│
├── src/                      # Source code for use in this project
│
├── models/                   # Trained model weights and checkpoints (Git-ignored)
│
├── outputs/                  # Generated plots, prediction samples, and logs
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project landing page & documentation
└── .gitignore                # Specifies intentionally untracked files to ignore
```

---

## 🛠️ Getting Started

### 1. Setup Environment
We recommend using a Python virtual environment (version 3.8+):

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries for computer vision, OCR, deep learning, and data analysis:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Key Modules & Directories

- **`data/`**: Keep data modular. Do not commit large binary dataset files directly to Git. Only the folder structure is tracked via `.gitkeep` files.
- **`notebooks/`**: Dedicated to exploratory data analysis (EDA), prototype models, and hyperparameter tuning. Keep notebooks named sequentially, e.g., `01_data_exploration.ipynb`.
- **`src/`**: Houses production-ready Python modules for data loading, preprocessing, model definitions, training loops, and evaluation metrics.
- **`models/`**: Stashes local checkpoints. Highly recommended to upload final model weights to a model registry (e.g., Hugging Face Hub, MLflow).
- **`outputs/`**: Any visual outputs, confusion matrices, prediction tables, or log files go here.

---

## ⚖️ License
This project is licensed under the MIT License.
