from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_PATH = Path("Data/raw/appliances_energy_raw.csv")
PROCESSED_PATH = Path("Data/processed/appliances_energy_course.csv")
METADATA_PATH = Path("Data/metadata/appliances_energy_feature_dictionary.csv")


# --------------------------------------------------
# Load raw data
# --------------------------------------------------

data = pd.read_csv(RAW_PATH)

print("Raw shape:", data.shape)


# --------------------------------------------------
# Validate raw dataset
# --------------------------------------------------

assert data.shape[0] == 19735, "Unexpected number of rows."
assert data.isna().sum().sum() == 0, "Unexpected missing values."
assert data.duplicated().sum() == 0, "Unexpected duplicate rows."

# In the downloaded source file rv1 and rv2 are exact duplicates.
assert (data["rv1"] == data["rv2"]).all(), \
    "rv1 and rv2 are no longer identical."


# --------------------------------------------------
# Rename features for teaching
# --------------------------------------------------

rename_map = {
    "date": "timestamp",
    "Appliances": "appliance_energy_wh",
    "lights": "lights_energy_wh",

    "T1": "kitchen_temp_c",
    "RH_1": "kitchen_humidity_pct",

    "T2": "living_room_temp_c",
    "RH_2": "living_room_humidity_pct",

    "T3": "laundry_room_temp_c",
    "RH_3": "laundry_room_humidity_pct",

    "T4": "office_temp_c",
    "RH_4": "office_humidity_pct",

    "T5": "bathroom_temp_c",
    "RH_5": "bathroom_humidity_pct",

    "T6": "north_outdoor_temp_c",
    "RH_6": "north_outdoor_humidity_pct",

    "T7": "ironing_room_temp_c",
    "RH_7": "ironing_room_humidity_pct",

    "T8": "teen_room_temp_c",
    "RH_8": "teen_room_humidity_pct",

    "T9": "parents_room_temp_c",
    "RH_9": "parents_room_humidity_pct",

    "T_out": "weather_temp_c",
    "Press_mm_hg": "weather_pressure_mm_hg",
    "RH_out": "weather_humidity_pct",
    "Windspeed": "weather_windspeed_ms",
    "Visibility": "weather_visibility_km",
    "Tdewpoint": "weather_dewpoint_c",

    "rv1": "random_control",
}

course = data.rename(columns=rename_map).copy()


# --------------------------------------------------
# Prepare timestamp
# --------------------------------------------------

course["timestamp"] = pd.to_datetime(course["timestamp"])

course = course.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------
# Add simple time features
# --------------------------------------------------

course["hour"] = course["timestamp"].dt.hour
course["day_of_week"] = course["timestamp"].dt.dayofweek
course["is_weekend"] = (
    course["day_of_week"] >= 5
).astype(int)


# --------------------------------------------------
# Remove duplicate random variable
# --------------------------------------------------

# rv2 is preserved in the untouched raw file.
# The course dataset keeps one random variable as a
# negative-control feature for model interpretation.
course = course.drop(columns=["rv2"])


# --------------------------------------------------
# Organize columns
# --------------------------------------------------

ordered_columns = [
    "timestamp",
    "appliance_energy_wh",

    "lights_energy_wh",

    "kitchen_temp_c",
    "kitchen_humidity_pct",
    "living_room_temp_c",
    "living_room_humidity_pct",
    "laundry_room_temp_c",
    "laundry_room_humidity_pct",
    "office_temp_c",
    "office_humidity_pct",
    "bathroom_temp_c",
    "bathroom_humidity_pct",
    "north_outdoor_temp_c",
    "north_outdoor_humidity_pct",
    "ironing_room_temp_c",
    "ironing_room_humidity_pct",
    "teen_room_temp_c",
    "teen_room_humidity_pct",
    "parents_room_temp_c",
    "parents_room_humidity_pct",

    "weather_temp_c",
    "weather_pressure_mm_hg",
    "weather_humidity_pct",
    "weather_windspeed_ms",
    "weather_visibility_km",
    "weather_dewpoint_c",

    "hour",
    "day_of_week",
    "is_weekend",

    "random_control",
]

course = course[ordered_columns]


# --------------------------------------------------
# Final validation
# --------------------------------------------------

assert course.isna().sum().sum() == 0
assert course.duplicated().sum() == 0
assert course["timestamp"].is_monotonic_increasing

print("Course shape:", course.shape)
print(
    "Date range:",
    course["timestamp"].min(),
    "to",
    course["timestamp"].max(),
)


# --------------------------------------------------
# Save course dataset
# --------------------------------------------------

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

course.to_csv(
    PROCESSED_PATH,
    index=False,
)


# --------------------------------------------------
# Feature dictionary
# --------------------------------------------------

feature_dictionary = [
    {
        "feature": "timestamp",
        "source_column": "date",
        "role": "time",
        "group": "time",
        "description": "Date and time of the 10-minute observation.",
        "unit": "",
    },
    {
        "feature": "appliance_energy_wh",
        "source_column": "Appliances",
        "role": "target",
        "group": "energy",
        "description": "Appliance energy use during the recorded interval.",
        "unit": "Wh",
    },
    {
        "feature": "lights_energy_wh",
        "source_column": "lights",
        "role": "feature",
        "group": "energy",
        "description": "Energy use of house light fixtures.",
        "unit": "Wh",
    },

    {
        "feature": "kitchen_temp_c",
        "source_column": "T1",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the kitchen area.",
        "unit": "C",
    },
    {
        "feature": "kitchen_humidity_pct",
        "source_column": "RH_1",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the kitchen area.",
        "unit": "%",
    },

    {
        "feature": "living_room_temp_c",
        "source_column": "T2",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the living-room area.",
        "unit": "C",
    },
    {
        "feature": "living_room_humidity_pct",
        "source_column": "RH_2",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the living-room area.",
        "unit": "%",
    },

    {
        "feature": "laundry_room_temp_c",
        "source_column": "T3",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the laundry-room area.",
        "unit": "C",
    },
    {
        "feature": "laundry_room_humidity_pct",
        "source_column": "RH_3",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the laundry-room area.",
        "unit": "%",
    },

    {
        "feature": "office_temp_c",
        "source_column": "T4",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the office room.",
        "unit": "C",
    },
    {
        "feature": "office_humidity_pct",
        "source_column": "RH_4",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the office room.",
        "unit": "%",
    },

    {
        "feature": "bathroom_temp_c",
        "source_column": "T5",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the bathroom.",
        "unit": "C",
    },
    {
        "feature": "bathroom_humidity_pct",
        "source_column": "RH_5",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the bathroom.",
        "unit": "%",
    },

    {
        "feature": "north_outdoor_temp_c",
        "source_column": "T6",
        "role": "feature",
        "group": "local_outdoor",
        "description": "Temperature outside the building on the north side.",
        "unit": "C",
    },
    {
        "feature": "north_outdoor_humidity_pct",
        "source_column": "RH_6",
        "role": "feature",
        "group": "local_outdoor",
        "description": "Relative humidity outside the building on the north side.",
        "unit": "%",
    },

    {
        "feature": "ironing_room_temp_c",
        "source_column": "T7",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the ironing room.",
        "unit": "C",
    },
    {
        "feature": "ironing_room_humidity_pct",
        "source_column": "RH_7",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the ironing room.",
        "unit": "%",
    },

    {
        "feature": "teen_room_temp_c",
        "source_column": "T8",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in teenager room 2.",
        "unit": "C",
    },
    {
        "feature": "teen_room_humidity_pct",
        "source_column": "RH_8",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in teenager room 2.",
        "unit": "%",
    },

    {
        "feature": "parents_room_temp_c",
        "source_column": "T9",
        "role": "feature",
        "group": "indoor_temperature",
        "description": "Temperature in the parents' room.",
        "unit": "C",
    },
    {
        "feature": "parents_room_humidity_pct",
        "source_column": "RH_9",
        "role": "feature",
        "group": "indoor_humidity",
        "description": "Relative humidity in the parents' room.",
        "unit": "%",
    },

    {
        "feature": "weather_temp_c",
        "source_column": "T_out",
        "role": "feature",
        "group": "weather",
        "description": "Outdoor temperature from Chievres weather station.",
        "unit": "C",
    },
    {
        "feature": "weather_pressure_mm_hg",
        "source_column": "Press_mm_hg",
        "role": "feature",
        "group": "weather",
        "description": "Atmospheric pressure from the weather station.",
        "unit": "mm Hg",
    },
    {
        "feature": "weather_humidity_pct",
        "source_column": "RH_out",
        "role": "feature",
        "group": "weather",
        "description": "Outdoor relative humidity from the weather station.",
        "unit": "%",
    },
    {
        "feature": "weather_windspeed_ms",
        "source_column": "Windspeed",
        "role": "feature",
        "group": "weather",
        "description": "Wind speed from the weather station.",
        "unit": "m/s",
    },
    {
        "feature": "weather_visibility_km",
        "source_column": "Visibility",
        "role": "feature",
        "group": "weather",
        "description": "Visibility from the weather station.",
        "unit": "km",
    },
    {
        "feature": "weather_dewpoint_c",
        "source_column": "Tdewpoint",
        "role": "feature",
        "group": "weather",
        "description": "Dew-point temperature from the weather station.",
        "unit": "C",
    },

    {
        "feature": "hour",
        "source_column": "derived from date",
        "role": "feature",
        "group": "time",
        "description": "Hour of day derived from timestamp.",
        "unit": "0-23",
    },
    {
        "feature": "day_of_week",
        "source_column": "derived from date",
        "role": "feature",
        "group": "time",
        "description": "Day of week derived from timestamp; Monday=0 and Sunday=6.",
        "unit": "0-6",
    },
    {
        "feature": "is_weekend",
        "source_column": "derived from date",
        "role": "feature",
        "group": "time",
        "description": "Weekend indicator derived from day of week.",
        "unit": "0/1",
    },

    {
        "feature": "random_control",
        "source_column": "rv1",
        "role": "negative_control",
        "group": "control",
        "description": (
            "Random variable included in the original dataset for testing "
            "whether models assign predictive value to meaningless inputs."
        ),
        "unit": "",
    },
]

metadata = pd.DataFrame(feature_dictionary)

assert set(metadata["feature"]) == set(course.columns)

METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

metadata.to_csv(
    METADATA_PATH,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nSaved:")
print(PROCESSED_PATH)
print(METADATA_PATH)

print("\nFinal columns:")
for column in course.columns:
    print("-", column) 

