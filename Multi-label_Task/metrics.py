import pandas as pd
import os

impact_columns = [
    "Infrastructural impact", 
    "Political impact", 
    "Economic impact", 
    "Ecological impact", 
    "Agricultural impact", 
    "Human health impact"
]
groupby=["Date","Time_Period"]
gold_data = pd.read_csv("the_path_to_gold_data.csv")
gold_data.columns = [x.capitalize() for x in gold_data.columns]

def eval_row_wise_acc(data, output_file):
    data.columns = [x.capitalize() for x in data.columns]
    models = data['Model_type'].unique()

    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()
    results = []

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()
        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        all_correct = (merged[[f"{col}_model" for col in impact_columns]].values ==
                       merged[[f"{col}_gold" for col in impact_columns]].values).all(axis=1)

        accuracy = all_correct.sum() / len(all_correct) if len(all_correct) > 0 else 0
        results.append({
            "Model_Type": model,
            "Row-Wise-Accuracy": round(accuracy, 4)
        })

    df_result = pd.DataFrame(results)
    if not os.path.isfile(output_file):
        df_result.to_csv(output_file, index=False)
    else:
        df_result.to_csv(output_file, mode='a', header=False, index=False)

def eval_metrics(data, output_file):
    data.columns = [x.capitalize() for x in data.columns]
    models = data["Model_type"].unique()
    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()
    results = []

    for model in models:
        model_data = data[data["Model_type"] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()
        merged = grouped.join(gold_grouped, how="inner", lsuffix="_model", rsuffix="_gold")

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

                metrics[col] = round(value, 4)
            results.append(metrics)

    df_result = pd.DataFrame(results)
    print(df_result)

    if not os.path.isfile(output_file):
        df_result.to_csv(output_file, index=False)
    else:
        df_result.to_csv(output_file, mode="a", header=False, index=False)

data = pd.read_csv("/content/output_gpt.csv")
eval_metrics(data, "accuracy_results.csv")