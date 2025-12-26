import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class CSVDataset(Dataset):
    def __init__(self, csv_path, project_root, class_to_idx, transform=None):
        df = pd.read_csv(csv_path)

        raw_files = df["file"].astype(str).tolist()
        self.paths = [os.path.normpath(os.path.join(project_root, f.replace("\\", "/"))) for f in raw_files]

        labels_raw = df["race"].astype(str).str.strip().str.lower().tolist()
        self.labels = [class_to_idx[lbl] for lbl in labels_raw]

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def extract_embeddings(loader, model, device, tag):
    model.eval()
    feats_list, y_list = [], []
    total = len(loader.dataset)
    done = 0

    with torch.inference_mode():
        for i, (x, y) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            feats = model(x)  # (B, 2048)
            feats_list.append(feats.cpu().numpy())
            y_list.append(np.asarray(y))

            done += x.size(0)
            if i % 20 == 0 or done == total:
                print(f"[{tag}] {done}/{total} images")

    X = np.concatenate(feats_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return X, y


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    ckpt_path = os.path.join(project_root, "models", "resnet50_finetuned_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (run finetune first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    print("Classes:", classes)

    # rebuild model and load weights
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.to(device)

    # convert to embedding extractor
    m.fc = nn.Identity()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_csv = os.path.join(project_root, "final Dataset", "cleaned_train.csv")
    val_csv   = os.path.join(project_root, "final Dataset", "cleaned_val.csv")

    train_ds = CSVDataset(train_csv, project_root, class_to_idx, transform=tf)
    val_ds   = CSVDataset(val_csv,   project_root, class_to_idx, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    out_dir = os.path.join(project_root, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    print("Extracting TRAIN embeddings (fine-tuned ResNet50)...")
    X_train, y_train = extract_embeddings(train_loader, m, device, tag="train")
    print("Train shape:", X_train.shape, y_train.shape)

    print("Extracting VAL embeddings (fine-tuned ResNet50)...")
    X_val, y_val = extract_embeddings(val_loader, m, device, tag="val")
    print("Val shape:", X_val.shape, y_val.shape)

    np.save(os.path.join(out_dir, "X_resnet50_finetuned_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_resnet50_finetuned_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_resnet50_finetuned_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_resnet50_finetuned_val.npy"), y_val)

    with open(os.path.join(out_dir, "classes_resnet50_finetuned.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    print("✅ Saved embeddings to:", out_dir)
    print(" - X_resnet50_finetuned_train.npy / y_resnet50_finetuned_train.npy")
    print(" - X_resnet50_finetuned_val.npy   / y_resnet50_finetuned_val.npy")
    print(" - classes_resnet50_finetuned.txt")


if __name__ == "__main__":
    main()
