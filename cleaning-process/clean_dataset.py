import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TRAIN_IN = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-train.csv")
VAL_IN   = os.path.join(PROJECT_ROOT, "initial Dataset", "ethnicity-classifier-val.csv")

FINAL_DIR = os.path.join(PROJECT_ROOT, "final Dataset")
os.makedirs(FINAL_DIR, exist_ok=True)

TRAIN_OUT = os.path.join(FINAL_DIR, "cleaned_train.csv")
VAL_OUT   = os.path.join(FINAL_DIR, "cleaned_val.csv")

# 🔧 Adjust these to match your CSV columns!
# Common ones: ["image_path", "label"] or ["path", "ethnicity"]
REQUIRED_COLS = None  # set to list if you want strict dropna(subset=...)

def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"\n📄 {name} original rows: {len(df)}")

    # 1) Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"🧹 {name}: removed {before - len(df)} duplicate rows")

    # 2) Drop null rows
    before = len(df)
    if REQUIRED_COLS is None:
        df = df.dropna()
        print(f"🧹 {name}: removed {before - len(df)} rows with ANY null")
    else:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: REQUIRED_COLS not found in CSV: {missing}")
        df = df.dropna(subset=REQUIRED_COLS)
        print(f"🧹 {name}: removed {before - len(df)} rows with nulls in {REQUIRED_COLS}")

    df = df.reset_index(drop=True)
    print(f"✅ {name} final rows: {len(df)}")
    return df

# === TRAIN ===
train_df = pd.read_csv(TRAIN_IN)
train_df = clean(train_df, "TRAIN")
train_df.to_csv(TRAIN_OUT, index=False)
print(f"💾 Saved: {TRAIN_OUT}")

# === VAL ===
val_df = pd.read_csv(VAL_IN)
val_df = clean(val_df, "VAL")
val_df.to_csv(VAL_OUT, index=False)
print(f"💾 Saved: {VAL_OUT}")

print("\n🎯 Cleaning complete.")
