import os
import pandas as pd
import re

# 创建存放结果的目录
output_dir = "./f1avg_result"
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
gold_data_350 = pd.read_csv("./final_query_annotated_350.csv")
gold_data_350.columns = [x.capitalize() for x in gold_data_350.columns]

gold_data_1300 = pd.read_csv("./final_query_annotated_1300.csv")
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

def evaluate_metrics(data, gold_data, model_name, category, version, shot):
    """ 计算 Micro-Averaged F1 Score 并保存 """
    data.columns = [x.capitalize() for x in data.columns]

    # 修正 Health impact -> Human health impact
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)

    gold_data_filtered = gold_data

    # 预处理 gold_data
    gold_grouped = gold_data_filtered.groupby(groupby)[impact_columns].max()

    # 计算评估指标
    results = []
    models = data['Model_type'].unique()
    
    # Initialize counters for micro-averaging
    total_tp = total_fp = total_fn = total_tn = 0

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()

        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')
        model_metrics = {"Model_Type": model_name, "Category": category, "Version": version, "Shot": shot, "Metric": "Micro-Averaged F1"}

        for col in impact_columns:
            tp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 1)).sum()
            tn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 0)).sum()
            fp = ((merged[f"{col}_model"] == 1) & (merged[f"{col}_gold"] == 0)).sum()
            fn = ((merged[f"{col}_model"] == 0) & (merged[f"{col}_gold"] == 1)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
        print(total_fn+total_fp+total_tp+total_tn)
        print(model)
        precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        f1_micro *= 100
        model_metrics["Micro-Averaged F1"] = round(f1_micro, 4)
        results.append(model_metrics)

    return results


all_results = []

# 遍历所有拆分后的文件
for file_path in csv_files:
    match = pattern.search(os.path.basename(file_path))
    if not match:
        print(f"Skipping {file_path}: 文件名格式不符合预期")
        continue
    print(file_path)
    model_name, category, version, shot = match.groups() # deepseek, historical/modern, 350/1300, oneshot/zeroshot
    model_full_name = f"{model_name}_{version}_{shot}"

    # 选择正确的 gold_data
    gold_data = gold_data_350 if version == "350" else gold_data_1300

    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 计算评估指标
    model_results = evaluate_metrics(data, gold_data, model_name, category, version, shot)
    all_results.extend(model_results)

# 将所有结果写入一个 CSV 文件
output_file = os.path.join(output_dir, "combined_metrics.csv")
df_result = pd.DataFrame(all_results)
df_result.to_csv(output_file, index=False)

print("所有评估已完成，结果存放在 ./f1avg_result/combined_metrics.csv 文件中！")
