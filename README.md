# Ethnicity Classification from Face Images  
**CNN Feature Extraction + Classical Machine Learning**

This repository contains our **CSCI441 – Machine Learning** course project.  
The objective of this project is to classify ethnicity from facial images using a **hybrid machine learning pipeline** that combines deep learning–based feature extraction with classical machine learning classifiers.

Rather than training an end-to-end deep learning classifier, this project focuses on:
- extracting strong visual features using a fine-tuned CNN
- comparing the performance of multiple classical ML models under identical conditions

---

## Project Workflow

1. **Dataset Cleaning**
   - Removal of missing values and duplicate samples
   - Generation of clean training and validation CSV files

2. **CNN Fine-Tuning**
   - Fine-tuning of a ResNet50 network on the cleaned training data
   - Saving the best-performing model checkpoint

3. **Feature Extraction**
   - Conversion of face images into fixed-length embedding vectors
   - Storage of embeddings as NumPy arrays

4. **Classical ML Training & Evaluation**
   - Training SVM, KNN, and Random Forest classifiers
   - Evaluation using accuracy, confusion matrices, and class-wise F1 scores

---

## Repository Structure

CSCI441-ETHNICITY-IMAGE-CLASSIFIER/
│
├── cleaning-process/ # Dataset validation and cleaning scripts
│ ├── check_duplicates.py
│ ├── check_nulls.py
│ └── clean_dataset.py
│
├── converting-scripts/ # # CNN fine-tuning and feature extraction (GPU-based)
│ ├── finetune_resnet50.py
│ └── extract_embeddings_finetuned.py
│
├── embeddings/ # Saved CNN feature embeddings
│ ├── classes_resnet50_finetuned.txt
│ ├── X_resnet50_finetuned_train.npy
│ ├── X_resnet50_finetuned_val.npy
│ ├── y_resnet50_finetuned_train.npy
│ └── y_resnet50_finetuned_val.npy
│
├── initial Dataset/ # Original dataset CSV files
│ ├── ethnicity-classifier-train.csv
│ └── ethnicity-classifier-val.csv
│
├── final Dataset/ # Cleaned dataset CSV files
│ ├── cleaned_train.csv
│ └── cleaned_val.csv
│
├── models/ # Saved CNN model checkpoint
│ └── resnet50_finetuned_best.pth
│
├── reports/figures/ # Evaluation figures used in the report
│ ├── cm_svm.png
│ ├── cm_knn.png
│ ├── cm_rf.png
│ └── classwise_f1_comparison.png
│
├── scripts/ # Classical ML training and evaluation scripts
│ ├── train_svm_classifier.py
│ ├── train_knn_classifier.py
│ ├── train_random_forest.py
│ ├── export_confusion_matrices.py
│ └── plot_classwise_f1.py
│
├── train/ # Image training folder (NOT pushed – large size)
├── val/ # Image validation folder (NOT pushed – large size)
│
├── svm_resnet50_finetuned.joblib # Generated after training (not pushed by default)
├── knn_resnet50_finetuned.joblib # Generated after training (not pushed by default)
├── rf_resnet50_finetuned.joblib # Generated after training (not pushed by default)
│
├── .gitignore
└── README.md


## Notes on Ignored Files and Folders

- The `train/` and `val/` directories contain raw image files and are **not pushed to GitHub** due to their large size.
- These folders must exist locally for fine-tuning and embedding extraction.
- The `.joblib` model files are **generated automatically** after running the classifier training scripts and are therefore not required to be version-controlled.

---

## CNN Fine-Tuning and Feature Extraction

A **ResNet50** model is fine-tuned using the cleaned training dataset.  
The fine-tuned network is then used **only as a feature extractor**, converting each image into a fixed-length embedding vector.

These embeddings are saved as NumPy arrays and reused across all classical ML experiments to ensure a **fair and consistent comparison**.

---

## Classical Machine Learning Models

Using the extracted embeddings, the following classifiers are trained and evaluated:

- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (KNN)**
- **Random Forest (RF)**

All models share the same embeddings, data splits, and preprocessing pipeline.

---

## Evaluation

Model performance is evaluated using:
- Validation accuracy
- Confusion matrices
- Class-wise F1 score comparison

All evaluation plots are stored in:
reports/figures/


---o

## How to Run the Project

1. Clean the dataset:
```bash
python cleaning-process/clean_dataset.py
Fine-tune the CNN:

bash
python converting-scripts/finetune_resnet50.py
Extract feature embeddings:

bash
python converting-scripts/extract_embeddings_finetuned.py
Train classical ML models:

bash
python scripts/train_svm_classifier.py
python scripts/train_knn_classifier.py
python scripts/train_random_forest.py
Generate evaluation figures:

bash
python scripts/export_confusion_matrices.py
python scripts/plot_classwise_f1.py
```


Course: CSCI441 – Machine Learning
Project Type: Course Project

Topic: Ethnicity Classification from Facial Images
Project Type: Course Project
Topic: Ethnicity Classification from Facial Images
