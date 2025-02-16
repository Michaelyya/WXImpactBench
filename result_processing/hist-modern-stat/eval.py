import os
import pandas as pd
import re

output_dir = "./result"
os.makedirs(output_dir, exist_ok=True)

# Define impact columns and groupby keys
impact_columns = [
    "Infrastructural impact",
    "Political impact",
    "Financial impact",
    "Ecological impact",
    "Agricultural impact",
    "Human health impact"
]
groupby = ['Date', 'Type', 'Id']

print(os.getcwd())

gold_data_350 = pd.read_csv("final_query_annotated_350.csv")
gold_data_350.columns = [x.capitalize() for x in gold_data_350.columns]

gold_data_1300 = pd.read_csv("final_query_annotated_1300.csv")
gold_data_1300.columns = [x.capitalize() for x in gold_data_1300.columns]

# Collect all CSV files in ./split directory
split_dir = "./split"
csv_files = []
for root, _, files in os.walk(split_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

# Regex pattern to extract details from filenames (e.g., "deepseek_historical_350_oneshot.csv")
pattern = re.compile(r"(.+?)_(historical|modern)_(\d+)_(oneshot|zeroshot)\.csv")

def evaluate_metrics(data, gold_data, model_name, category, version, shot, output_file):
    """ Compute Precision, Recall, F1, and Accuracy. """
    data.columns = [x.capitalize() for x in data.columns]

    # Ensure column consistency
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)

    # Filter gold data to match historical/modern type
    gold_data_filtered = gold_data[gold_data["Type"] == category]
    gold_grouped = gold_data_filtered.groupby(groupby)[impact_columns].max()

    results = []
    models = data['Model_type'].unique()

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()
        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        for metric_name in ["Precision", "Recall", "F1", "Accuracy"]:
            metrics = {"Model_Type": model_name, "Type": category, "Version": version, "Shot": shot, "Metric": metric_name}

            for col in impact_columns:
                tp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 1)).sum()
                tn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 0)).sum()
                fp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 0)).sum()
                fn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 1)).sum()

                if metric_name == "Precision":
                    value = tp / (tp + fp) if (tp + fp) > 0 else 0
                elif metric_name == "Recall":
                    value = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric_name == "F1":
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                elif metric_name == "Accuracy":
                    value = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

                value *= 100
                metrics[col] = round(value, 4)
            results.append(metrics)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)

# Process all CSV files in ./split
for file_path in csv_files:
    match = pattern.search(os.path.basename(file_path))
    if not match:
        print(f"Skipping {file_path}: Invalid filename format")
        continue

    model_name, category, version, shot = match.groups()  # Extract model details from filename
    model_full_name = f"{model_name}_{category}_{version}_{shot}"

    # Select correct ground truth data
    gold_data = gold_data_350 if version == "350" else gold_data_1300

    data = pd.read_csv(file_path)
    output_file = os.path.join(output_dir, f"{model_full_name}_metrics.csv")
    evaluate_metrics(data, gold_data, model_name, category, version, shot, output_file)

print("Evaluation completed. Results are saved in ./result directory.")
