import os, copy, time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class CSVDataset(Dataset):
    """
    CSV must have columns: file, race
    file: path relative to PROJECT ROOT (e.g., train/xxx.jpg)
    race: label string
    """
    def __init__(self, csv_path, project_root, class_to_idx=None, transform=None):
        df = pd.read_csv(csv_path)

        raw_files = df["file"].astype(str).tolist()
        self.paths = [os.path.normpath(os.path.join(project_root, f.replace("\\", "/"))) for f in raw_files]

        labels_raw = df["race"].astype(str).str.strip().str.lower().tolist()
        if class_to_idx is None:
            classes = sorted(set(labels_raw))
            class_to_idx = {c: i for i, c in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        self.labels = [class_to_idx[lbl] for lbl in labels_raw]

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def main():
    # Robust project root (works even if you run from converting-scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    train_csv = os.path.join(project_root, "final Dataset", "cleaned_train.csv")
    val_csv   = os.path.join(project_root, "final Dataset", "cleaned_val.csv")

    batch_size = 64
    epochs = 6
    lr = 1e-4
    wd = 1e-4
    num_workers = 0  # Windows-safe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CSVDataset(train_csv, project_root, class_to_idx=None, transform=train_tf)
    val_ds   = CSVDataset(val_csv,   project_root, class_to_idx=train_ds.class_to_idx, transform=val_tf)
    print("Classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))

    # Light fine-tune: layer4 + fc
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    model.to(device)

    # Balanced loss
    counts = np.bincount(np.array(train_ds.labels), minlength=len(train_ds.classes)).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def eval_val():
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total

    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = eval_val()

        print(f"Epoch {ep}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{(time.time()-t0)/60:.1f} min")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print("✅ New best, saved in memory")

    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    ckpt_path = os.path.join(project_root, "models", "resnet50_finetuned_best.pth")
    torch.save({
        "state_dict": best_state,
        "class_to_idx": train_ds.class_to_idx,
        "classes": train_ds.classes
    }, ckpt_path)

    print("💾 Saved:", ckpt_path)
    print("🏁 Best val_acc:", best_acc)


if __name__ == "__main__":
    main()
