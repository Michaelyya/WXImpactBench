import os
import pandas as pd
import re


output_dir = "./split"
os.makedirs(output_dir, exist_ok=True)

csv_files = []
for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))  # Full path

# Regex to extract parts like "350_oneshot" or "1300_zeroshot"
pattern = re.compile(r"(\d+_(?:oneshot|zeroshot))")

for file_path in csv_files:
    match = pattern.search(file_path)
    if not match:
        continue

    file_suffix = match.group(1)  # Extract "350_oneshot" or "1300_zeroshot"

    df = pd.read_csv(file_path)

    # Ensure required columns exist
    if "Type" not in df.columns:
        print(f"Skipping {file_path}: No 'Type' column.")
        continue

    if "Model_Type" not in df.columns:
        print(f"Skipping {file_path}: No 'Model_Type' column.")
        continue

    # Split data by "Type"
    historical_df = df[df["Type"] == "historical"]
    modern_df = df[df["Type"] == "modern"]

    base_model_name = "_".join(os.path.basename(file_path).split("_")[:-2])

    if not historical_df.empty:
        historical_output_path = os.path.join(output_dir, f"{base_model_name}_historical_{file_suffix}.csv")
        historical_df.to_csv(historical_output_path, index=False)

    if not modern_df.empty:
        modern_output_path = os.path.join(output_dir, f"{base_model_name}_modern_{file_suffix}.csv")
        modern_df.to_csv(modern_output_path, index=False)

print("Splitting complete. All files saved in ./split directory.")
