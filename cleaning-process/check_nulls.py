import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

train_csv = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-train.csv")
val_csv   = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-val.csv")

backup_dir = os.path.join(PROJECT_ROOT, "backups")
os.makedirs(backup_dir, exist_ok=True)

# === TRAIN ===
df_train = pd.read_csv(train_csv)

null_counts_train = df_train.isnull().sum()
null_summary_train = null_counts_train[null_counts_train > 0]

if not null_summary_train.empty:
    print("\n📊 Nulls found in the following columns (TRAIN):")
    print(null_summary_train)

    null_rows_train = df_train[df_train.isnull().any(axis=1)]
    out_train = os.path.join(backup_dir, "null_rows_backup_train.csv")
    null_rows_train.to_csv(out_train, index=False)
    print(f"💾 Rows with nulls saved to: {out_train}")
else:
    print("✅ No nulls found in any column (TRAIN).")

# === VAL ===
df_val = pd.read_csv(val_csv)

null_counts_val = df_val.isnull().sum()
null_summary_val = null_counts_val[null_counts_val > 0]

if not null_summary_val.empty:
    print("\n📊 Nulls found in the following columns (VAL):")
    print(null_summary_val)

    null_rows_val = df_val[df_val.isnull().any(axis=1)]
    out_val = os.path.join(backup_dir, "null_rows_backup_val.csv")
    null_rows_val.to_csv(out_val, index=False)
    print(f"💾 Rows with nulls saved to: {out_val}")
else:
    print("✅ No nulls found in any column (VAL).")
