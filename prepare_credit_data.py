from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Download the official dataset
credit_dataset = fetch_ucirepo(id=350)

# Keep the complete original table:
# ID + X1-X23 + Y
raw_data = credit_dataset.data.original.copy()

# Create destination folder
output_folder = Path("Data/raw")
output_folder.mkdir(parents=True, exist_ok=True)

# Save true raw copy
output_path = output_folder / "credit_card_default_raw.csv"

raw_data.to_csv(output_path, index=False)

print("Raw dataset saved successfully.")
print("Shape:", raw_data.shape)
print("Columns:", raw_data.columns.tolist())