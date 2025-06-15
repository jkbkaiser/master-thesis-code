from pathlib import Path

import pandas as pd

# Paths
HF_DATA_DIR = Path("data/hf")
metadata_path = HF_DATA_DIR / "metadata.csv"
image_dir = HF_DATA_DIR / "images"
image_column = "file_name"  # Update if named differently

# Load metadata
df = pd.read_csv(metadata_path)

# Extract just the filename from ./images/uuid.webp
df["base_name"] = df[image_column].apply(lambda x: Path(x).name)

# Get actual files on disk
all_files_on_disk = {f.name for f in image_dir.iterdir() if f.is_file()}
valid_files_in_metadata = set(df["base_name"])

# 1. Remove extra files from disk
extra_files = all_files_on_disk - valid_files_in_metadata
print(f"Found {len(extra_files)} extra image files. Removing...")

for fname in extra_files:
    try:
        (image_dir / fname).unlink()
    except Exception as e:
        print(f"Failed to delete {fname}: {e}")

existing_files = all_files_on_disk
filtered_df = df[df["base_name"].isin(existing_files)]

print(f"Filtered metadata: {len(df)} → {len(filtered_df)} rows")

# Drop helper column and save
filtered_df.drop(columns=["base_name"], inplace=True)
filtered_df.to_csv(HF_DATA_DIR / "metadata.cleaned.csv", index=False)
print("Saved cleaned metadata to metadata.cleaned.csv")
