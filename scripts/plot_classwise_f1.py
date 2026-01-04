import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_classwise_f1(model, X, y, classes):
    preds = model.predict(X)
    report = classification_report(
        y, preds, target_names=classes, output_dict=True, zero_division=0
    )

    # Extract F1-score for each class (in order)
    return [report[c]["f1-score"] for c in classes]


def main():
    # scripts/ -> project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    emb_dir = os.path.join(project_root, "embeddings")

    # Load validation data
    X_val = np.load(os.path.join(emb_dir, "X_resnet50_finetuned_val.npy"))
    y_val = np.load(os.path.join(emb_dir, "y_resnet50_finetuned_val.npy")).ravel()
    classes = load_classes(os.path.join(emb_dir, "classes_resnet50_finetuned.txt"))

    # Load trained models
    models = {
        "SVM": joblib.load(os.path.join(project_root, "svm_resnet50_finetuned.joblib")),
        "KNN": joblib.load(os.path.join(project_root, "knn_resnet50_finetuned.joblib")),
        "RF":  joblib.load(os.path.join(project_root, "rf_resnet50_finetuned.joblib")),
    }

    # Compute F1-scores
    f1_scores = {
        name: get_classwise_f1(model, X_val, y_val, classes)
        for name, model in models.items()
    }

    # ---- Plot (grouped bars) ----
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(x - width, f1_scores["SVM"], width, label="SVM")
    ax.bar(x,         f1_scores["KNN"], width, label="KNN")
    ax.bar(x + width, f1_scores["RF"],  width, label="Random Forest")

    ax.set_ylabel("F1-score")
    ax.set_xlabel("Ethnicity class")
    ax.set_title("Class-wise F1-score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Save figure
    out_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "classwise_f1_comparison.png")

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Saved class-wise F1 plot to: {out_path}")


if __name__ == "__main__":
    main()
