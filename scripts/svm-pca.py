import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
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

# === Apply PCA ===
pca = PCA(n_components=256)   # try 128, 256, 512 for tuning
X_train_pca = pca.fit_transform(X_train)
X_val_pca   = pca.transform(X_val)

print("After PCA:", X_train_pca.shape, X_val_pca.shape)

# === Train Linear SVM (baseline) ===
svm_linear = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
print("Training Linear SVM...")
svm_linear.fit(X_train_pca, y_train)

print("Evaluating Linear SVM...")
y_pred_linear = svm_linear.predict(X_val_pca)
print("Validation Accuracy (Linear):", accuracy_score(y_val, y_pred_linear))
print("\nClassification Report (Linear):\n", classification_report(y_val, y_pred_linear))

# === Confusion Matrix (Linear) ===
cm = confusion_matrix(y_val, y_pred_linear)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Linear SVM")
plt.show()

# === Save Linear model ===
joblib.dump(svm_linear, "svm_ethnicity_model_resnet50_pca_linear.pkl")
print("Linear SVM model saved.")

# === Train RBF SVM on a larger subset ===
n = 40000   # <-- increase subset size here (try 40k, 60k, etc.)
X_sub = X_train_pca[:n]
y_sub = y_train[:n]

svm_rbf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
print(f"\nTraining RBF SVM on {n} samples...")
svm_rbf.fit(X_sub, y_sub)

print("Evaluating RBF SVM (subset)...")
y_pred_rbf = svm_rbf.predict(X_val_pca)
print("Validation Accuracy (RBF subset):", accuracy_score(y_val, y_pred_rbf))
print("\nClassification Report (RBF subset):\n", classification_report(y_val, y_pred_rbf))

# === Save RBF model (subset) ===
joblib.dump(svm_rbf, f"svm_ethnicity_model_resnet50_pca_rbf_subset_{n}.pkl")
print(f"RBF SVM (subset {n}) model saved.")
