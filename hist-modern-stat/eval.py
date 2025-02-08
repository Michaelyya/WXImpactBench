import os
import pandas as pd
import re

# 创建存放结果的目录
output_dir = "./result"
os.makedirs(output_dir, exist_ok=True)

# 设置 impact_columns 和 gold_data（用于计算评估指标）
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
# 读取基准数据（注意需要 350 和 1300 两种数据集）
gold_data_350 = pd.read_csv("final_query_annotated_350.csv")
gold_data_350.columns = [x.capitalize() for x in gold_data_350.columns]

gold_data_1300 = pd.read_csv("final_query_annotated_1300.csv")
gold_data_1300.columns = [x.capitalize() for x in gold_data_1300.columns]

# 遍历 ./split/ 目录下的所有 CSV 文件
split_dir = "./split"
csv_files = []
for root, _, files in os.walk(split_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

# 正则匹配文件名中的信息，例如 "deepseek_historical_350_oneshot.csv"
pattern = re.compile(r"(.+?)_(historical|modern)_(\d+)_(oneshot|zeroshot)\.csv")

def evaluate_metrics(data, gold_data, model_name, category, output_file):
    """ 计算 Precision, Recall, F1, Accuracy 并保存 """
    data.columns = [x.capitalize() for x in data.columns]

    # 修正 Health impact -> Human health impact
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)

    # 按照 historical/modern 过滤 gold_data，确保评估数据一致
    gold_data_filtered = gold_data[gold_data["Type"] == category]

    # 预处理 gold_data
    gold_grouped = gold_data_filtered.groupby(groupby)[impact_columns].max()

    # 计算评估指标
    results = []
    models = data['Model_type'].unique()
    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()

        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        for metric_name in ["Precision", "Recall", "F1", "Accuracy"]:
            metrics = {"Model_Type": model_name, "Metric": metric_name}
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

    # 保存到 CSV
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)


# 遍历所有拆分后的文件
for file_path in csv_files:
    match = pattern.search(os.path.basename(file_path))
    if not match:
        print(f"Skipping {file_path}: 文件名格式不符合预期")
        continue

    model_name, category, version, shot = match.groups()  # deepseek, historical/modern, 350/1300, oneshot/zeroshot
    model_full_name = f"{model_name}_{category}_{version}_{shot}"

    # 选择正确的 gold_data
    gold_data = gold_data_350 if version == "350" else gold_data_1300

    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 结果文件路径
    output_file = os.path.join(output_dir, f"{model_full_name}_metrics.csv")

    # 计算评估指标
    evaluate_metrics(data, gold_data, model_name, category, output_file)

print("所有评估已完成，结果存放在 ./result 目录下！")
