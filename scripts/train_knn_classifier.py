import os
import time
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_classes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # scripts/ -> project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    emb_dir = os.path.join(project_root, "embeddings")

    # ✅ Fine-tuned embeddings
    X_train = np.load(os.path.join(emb_dir, "X_resnet50_finetuned_train.npy"))
    y_train = np.load(os.path.join(emb_dir, "y_resnet50_finetuned_train.npy"))
    X_val   = np.load(os.path.join(emb_dir, "X_resnet50_finetuned_val.npy"))
    y_val   = np.load(os.path.join(emb_dir, "y_resnet50_finetuned_val.npy"))
    classes = load_classes(os.path.join(emb_dir, "classes_resnet50_finetuned.txt"))

    print("✅ Loaded fine-tuned embeddings")
    print(f"Train: {X_train.shape} {y_train.shape} | dtype: {X_train.dtype} {y_train.dtype}")
    print(f"Val:   {X_val.shape} {y_val.shape} | dtype: {X_val.dtype} {y_val.dtype}")

    print("\n📌 Class index mapping:")
    for i, c in enumerate(classes):
        print(f"  {i} -> {c}")

    # 🔁 KNN classifier (same pipeline structure)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,       # common baseline
            weights="distance",  # improves performance on embeddings
            metric="euclidean",
            n_jobs=-1
        ))
    ])

    print("\n🧠 Training KNN...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"✅ Training done in {time.time() - t0:.1f} seconds")

    print("🔎 Evaluating on validation set...")
    t1 = time.time()
    preds = model.predict(X_val)
    print(f"✅ Prediction done in {time.time() - t1:.1f} seconds")

    acc = accuracy_score(y_val, preds)
    print(f"\n📌 Validation Accuracy: {acc:.4f}")

    print("\n📊 Classification Report:\n")
    print(classification_report(y_val, preds, target_names=classes))

    print("🧩 Confusion Matrix:\n")
    print(confusion_matrix(y_val, preds))

    out_path = os.path.join(project_root, "knn_resnet50_finetuned.joblib")
    joblib.dump(model, out_path)
    print(f"\n💾 Saved model to: {out_path}")


if __name__ == "__main__":
    main()
