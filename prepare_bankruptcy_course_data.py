from pathlib import Path
import pandas as pd


RAW_PATH = Path("Data/raw/taiwanese_bankruptcy_raw.csv")
OUT_PATH = Path("Data/processed/taiwanese_bankruptcy_course.csv")


# Load raw data
data = pd.read_csv(RAW_PATH)

# Clean column names from the original UCI file
data.columns = data.columns.str.strip()

# Basic validation
assert data.shape == (6819, 96), f"Unexpected raw shape: {data.shape}"
assert "Bankrupt?" in data.columns
assert data["Bankrupt?"].isin([0, 1]).all()
assert data.isna().sum().sum() == 0
assert data.duplicated().sum() == 0

# Remove constant feature
assert data["Net Income Flag"].nunique() == 1
data = data.drop(columns=["Net Income Flag"])

# Save course-ready dataset
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Processed shape:", data.shape)
print("Bankrupt cases:", int(data["Bankrupt?"].sum()))
print("Bankruptcy rate:", f"{data['Bankrupt?'].mean():.2%}")
