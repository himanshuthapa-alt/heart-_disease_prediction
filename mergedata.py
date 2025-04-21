import pandas as pd
import os

# Path where your data files are extracted
data_path = "/home/lucid/heart_dises_tracker/heart_disease_data"  # Replace with your actual path

# All dataset filenames and their respective country
file_country_map = {
    "processed.cleveland.data": "USA",
    "processed.hungarian.data": "Hungary",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "VA"
}

# Column names from UCI
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# List to hold cleaned DataFrames
cleaned_dfs = []

# Loop through files and clean/label
for filename, country in file_country_map.items():
    filepath = os.path.join(data_path, filename)
    df = pd.read_csv(filepath, names=columns)
    df['country'] = country

    # Replace '?' with NaN and convert to numeric
    df['ca'] = pd.to_numeric(df['ca'].replace('?', pd.NA))
    df['thal'] = pd.to_numeric(df['thal'].replace('?', pd.NA))

    cleaned_dfs.append(df)

# Merge all dataframes
merged_df = pd.concat(cleaned_dfs, ignore_index=True)

# Drop rows with missing values
merged_df.dropna(inplace=True)

# Convert target column to binary (0 = no disease, 1 = has disease)
merged_df['target'] = merged_df['target'].apply(lambda x: 1 if x > 0 else 0)

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Create output filename with timestamp
output_filename = "cleaned_heart_disease_data.csv"
output_path = os.path.join(data_path, output_filename)

# Save to CSV
merged_df.to_csv(output_path, index=False)

print("\nProcessing complete!")
print("Final dataset shape:", merged_df.shape)
print("Samples per country:\n", merged_df['country'].value_counts())
print(f"\nCleaned data saved to: {output_path}")