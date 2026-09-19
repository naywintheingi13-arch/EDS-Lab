from pathlib import Path

import pandas as pd


# -----------------------------------------
# 1. Load the untouched raw dataset
# -----------------------------------------

RAW_DATA_PATH = Path("Data/raw/credit_card_default_raw.csv")

data = pd.read_csv(RAW_DATA_PATH)


# -----------------------------------------
# 2. Rename columns to readable names
# -----------------------------------------

column_names = {
    "ID": "customer_id",
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education",
    "X4": "marital_status",
    "X5": "age",

    "X6": "payment_status_sep",
    "X7": "payment_status_aug",
    "X8": "payment_status_jul",
    "X9": "payment_status_jun",
    "X10": "payment_status_may",
    "X11": "payment_status_apr",

    "X12": "bill_amount_sep",
    "X13": "bill_amount_aug",
    "X14": "bill_amount_jul",
    "X15": "bill_amount_jun",
    "X16": "bill_amount_may",
    "X17": "bill_amount_apr",

    "X18": "payment_amount_sep",
    "X19": "payment_amount_aug",
    "X20": "payment_amount_jul",
    "X21": "payment_amount_jun",
    "X22": "payment_amount_may",
    "X23": "payment_amount_apr",

    "Y": "default_next_month"
}

data = data.rename(columns=column_names)


# -----------------------------------------
# 3. Group undocumented category codes
# -----------------------------------------

data["education"] = data["education"].replace({
    0: "Unknown",
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Other",
    5: "Unknown",
    6: "Unknown"
})

data["marital_status"] = data["marital_status"].replace({
    0: "Unknown",
    1: "Married",
    2: "Single",
    3: "Other"
})

data["sex"] = data["sex"].replace({
    1: "Male",
    2: "Female"
})


# -----------------------------------------
# 4. Save the student-ready dataset
# -----------------------------------------

OUTPUT_PATH = Path(
    "Data/processed/credit_card_default_course.csv"
)

data.to_csv(OUTPUT_PATH, index=False)

print("Course dataset saved successfully.")
print("Shape:", data.shape)
print(data.head())