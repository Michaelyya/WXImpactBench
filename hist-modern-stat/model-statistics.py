import os
import pandas as pd

analyze_type="long" #改这个！如果是350版本，用long; 如果是1300版本，用short


impact_columns = [
    "Infrastructural impact",
    "Political impact",
    "Financial impact",
    "Ecological impact",
    "Agricultural impact",
    "Human health impact"
]
groupby = ['Date', 'Type','Id']
gold_data = pd.read_csv(f"/content/{analyze_type}_context_labelled_y.csv")
gold_data.columns = [x.capitalize() for x in gold_data.columns]

csv_files=["/content/gemma-2-27b-350_zeroshot.csv"]
directory = "/content"
output_file = "/content/row-wise_accuracy.csv"

def evaluate_metrics(data, output_file):
    data.columns = [x.capitalize() for x in data.columns]
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)
    models = data['Model_type'].unique()
    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()
    results = []

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()

        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        for metric_name in ["Precision", "Recall", "F1", "Accuracy"]:
            metrics = {"Model_Type": model, "Metric": metric_name}
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

                metrics[col] = round(value,4)
            results.append(metrics)

    df_result = pd.DataFrame(results)
    print(df_result)
    df_result.to_csv(output_file, index=False)
    csv_string = df_result.to_csv(index=False)
    print(csv_string)

data = pd.read_csv("/deepseek-v3/gemma-2-27b-350_oneshot.csv")
evaluate_metrics(data, "metrics_results.csv")