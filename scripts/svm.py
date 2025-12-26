import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# === Load embeddings ===
X_train = np.load("embeddings/X_resnet50_train.npy")
y_train = np.load("embeddings/y_resnet50_train.npy")
X_val   = np.load("embeddings/X_resnet50_val.npy")
y_val   = np.load("embeddings/y_resnet50_val.npy")

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# === Scale features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# === Use subset of training data ===
n = 60000   # <-- subset size
X_sub = X_train[:n]
y_sub = y_train[:n]
print(f"Training on subset of {n} samples out of {X_train.shape[0]}")

# === Train Linear SVM on subset ===
svm_linear = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
print("Training Linear SVM (60k subset)...")
svm_linear.fit(X_sub, y_sub)

print("Evaluating Linear SVM (60k subset)...")
y_pred_linear = svm_linear.predict(X_val)
print("Validation Accuracy (Linear, 60k subset):", accuracy_score(y_val, y_pred_linear))
print("\nClassification Report (Linear, 60k subset):\n", classification_report(y_val, y_pred_linear))

# === Confusion Matrix (Linear) ===
cm = confusion_matrix(y_val, y_pred_linear)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Linear SVM (60k subset)")
plt.show()

# === Save Linear model ===
joblib.dump(svm_linear, "svm_ethnicity_model_resnet50_linear_60k.pkl")
print("Linear SVM (60k subset) model saved.")

# === Train RBF SVM on same subset ===
svm_rbf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
print("\nTraining RBF SVM (60k subset)...")
svm_rbf.fit(X_sub, y_sub)

print("Evaluating RBF SVM (60k subset)...")
y_pred_rbf = svm_rbf.predict(X_val)
print("Validation Accuracy (RBF, 60k subset):", accuracy_score(y_val, y_pred_rbf))
print("\nClassification Report (RBF, 60k subset):\n", classification_report(y_val, y_pred_rbf))

# === Save RBF model ===
joblib.dump(svm_rbf, "svm_ethnicity_model_resnet50_rbf_60k.pkl")
print("RBF SVM (60k subset) model saved.")
