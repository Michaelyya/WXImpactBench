import os
import pandas as pd
import re

output_dir = "./f1avg_result"
os.makedirs(output_dir, exist_ok=True)

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

# Load ground truth datasets (350 and 1300 versions)
gold_data_350 = pd.read_csv("./final_query_annotated_350.csv")
gold_data_350.columns = [x.capitalize() for x in gold_data_350.columns]

gold_data_1300 = pd.read_csv("./final_query_annotated_1300.csv")
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

def evaluate_metrics(data, gold_data, model_name, category, version, shot):
    """ Compute Micro-Averaged F1 Score. """
    data.columns = [x.capitalize() for x in data.columns]

    # Ensure column consistency
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)

    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()

    results = []
    models = data['Model_type'].unique()
    
    # Initialize counters for micro-averaging
    total_tp = total_fp = total_fn = total_tn = 0

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()
        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        model_metrics = {
            "Model_Type": model_name, "Category": category,
            "Version": version, "Shot": shot, "Metric": "Micro-Averaged F1"
        }

        for col in impact_columns:
            tp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 1)).sum()
            tn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 0)).sum()
            fp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 0)).sum()
            fn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 1)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        print(total_fn + total_fp + total_tp + total_tn)
        print(model)

        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

        model_metrics["Micro-Averaged F1"] = round(f1_micro * 100, 4)
        results.append(model_metrics)

    return results

all_results = []

# Process all CSV files in ./split
for file_path in csv_files:
    match = pattern.search(os.path.basename(file_path))
    if not match:
        print(f"Skipping {file_path}: Invalid filename format")
        continue

    print(file_path)
    model_name, category, version, shot = match.groups()
    model_full_name = f"{model_name}_{version}_{shot}"

    gold_data = gold_data_350 if version == "350" else gold_data_1300
    data = pd.read_csv(file_path)

    # Compute evaluation metrics
    model_results = evaluate_metrics(data, gold_data, model_name, category, version, shot)
    all_results.extend(model_results)

# Save all results to a single CSV file
output_file = os.path.join(output_dir, "combined_metrics.csv")
df_result = pd.DataFrame(all_results)
df_result.to_csv(output_file, index=False)

print("Evaluation completed. Results are saved in ./f1avg_result/combined_metrics.csv")
