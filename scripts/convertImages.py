import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# === Dataset class ===
class FairFaceCSV(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        df = pd.read_csv(csv_path)
        raw_files = df["file"].astype(str)
        fixed_files = []
        for fname in raw_files:
            fname = fname.replace("\\", "/")
            if fname.startswith(img_root + "/"):
                fname = fname[len(img_root) + 1:]
            fixed_files.append(os.path.join(img_root, fname))
        self.paths = fixed_files
        self.labels_raw = df["race"].astype(str).str.strip().str.lower()
        self.classes = sorted(set(self.labels_raw))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[lbl] for lbl in self.labels_raw]
        self.transform = transform

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.labels[idx]

# === Image transforms ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Embedding extraction with progress tracking ===
def extract_embeddings(dataloader, model, device, prefix):
    feats_list, labels_list = [], []
    model.eval()
    total = len(dataloader.dataset)
    processed = 0
    with torch.inference_mode():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats_list.append(feats.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            processed += len(imgs)
            percent = (processed / total) * 100
            print(f"[{prefix}] Batch {i+1}/{len(dataloader)} | Processed {processed}/{total} images ({percent:.2f}%)")
            print(f"[{prefix}] Example file:", dataloader.dataset.paths[processed-1])
    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

def save_embeddings(X, y, classes, prefix):
    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/X_{prefix}.npy", X)
    np.save(f"embeddings/y_{prefix}.npy", y)
    with open(f"embeddings/classes_{prefix}.txt", "w") as f:
        for c in classes: 
            f.write(c + "\n")
    print(f"Saved: embeddings/X_{prefix}.npy, y_{prefix}.npy, classes_{prefix}.txt")

# === Main block ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load pretrained ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet.to(device)

    # --- TRAIN SET ---
    csv_path_train = "../final Dataset/cleaned_train.csv"
    img_root_train = ".."
    train_ds = FairFaceCSV(csv_path_train, img_root_train, transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)

    print("Extracting TRAIN embeddings...")
    X_train, y_train = extract_embeddings(train_loader, resnet, device, prefix="train")
    print("Train embeddings shape:", X_train.shape)
    save_embeddings(X_train, y_train, train_ds.classes, prefix="resnet50_train")

    # --- VALIDATION SET ---
    csv_path_val = "../final Dataset/cleaned_val.csv"
    img_root_val = ".."
    val_ds = FairFaceCSV(csv_path_val, img_root_val, transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print("Extracting VAL embeddings...")
    X_val, y_val = extract_embeddings(val_loader, resnet, device, prefix="val")
    print("Val embeddings shape:", X_val.shape)
    save_embeddings(X_val, y_val, val_ds.classes, prefix="resnet50_val")
