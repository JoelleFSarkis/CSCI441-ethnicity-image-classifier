import os
import pandas as pd

# Run from cleaning-process/ (recommended)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

train_csv = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-train.csv")
val_csv   = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-val.csv")

backup_dir = os.path.join(PROJECT_ROOT, "backups")
os.makedirs(backup_dir, exist_ok=True)

# === Load CSVs ===
df_train = pd.read_csv(train_csv)
df_val   = pd.read_csv(val_csv)

# === Find duplicates in TRAIN ===
duplicates_train = df_train[df_train.duplicated()]
print(f"\n📎 Found {duplicates_train.shape[0]} duplicate rows in TRAIN")

if not duplicates_train.empty:
    out_train = os.path.join(backup_dir, "duplicates_backup_train.csv")
    duplicates_train.to_csv(out_train, index=False)
    print(f"💾 Saved TRAIN duplicates to: {out_train}")
else:
    print("✅ No duplicates found in TRAIN.")

# === Find duplicates in VAL ===
duplicates_val = df_val[df_val.duplicated()]
print(f"\n📎 Found {duplicates_val.shape[0]} duplicate rows in VAL")

if not duplicates_val.empty:
    out_val = os.path.join(backup_dir, "duplicates_backup_val.csv")
    duplicates_val.to_csv(out_val, index=False)
    print(f"💾 Saved VAL duplicates to: {out_val}")
else:
    print("✅ No duplicates found in VAL.")
