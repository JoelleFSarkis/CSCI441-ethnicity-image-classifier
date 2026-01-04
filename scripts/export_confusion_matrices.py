import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_confusion_matrix(y_true, y_pred, classes, title, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        cmap="Blues",          # clean light blue
        colorbar=False,
        xticks_rotation=45,
        values_format="d"
    )

    # Smaller, report-friendly fonts
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Smaller numbers inside the cells
    for text in disp.text_.ravel():
        text.set_fontsize(7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    # scripts/ -> project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    emb_dir = os.path.join(project_root, "embeddings")

    # Load validation data
    X_val = np.load(os.path.join(emb_dir, "X_resnet50_finetuned_val.npy"))
    y_val = np.load(os.path.join(emb_dir, "y_resnet50_finetuned_val.npy")).ravel()
    classes = load_classes(os.path.join(emb_dir, "classes_resnet50_finetuned.txt"))

    # Output directory
    out_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(out_dir, exist_ok=True)

    models = {
        "SVM": os.path.join(project_root, "svm_resnet50_finetuned.joblib"),
        "KNN": os.path.join(project_root, "knn_resnet50_finetuned.joblib"),
        "RF":  os.path.join(project_root, "rf_resnet50_finetuned.joblib"),
    }

    for name, path in models.items():
        print(f"🔎 Processing {name}...")
        model = joblib.load(path)
        preds = model.predict(X_val)

        out_path = os.path.join(out_dir, f"cm_{name.lower()}.png")
        save_confusion_matrix(
            y_true=y_val,
            y_pred=preds,
            classes=classes,
            title=f"{name} Confusion Matrix",
            out_path=out_path
        )

        print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    main()
