import os
import pandas as pd

# =========== 全局配置 ===========
impact_columns = [
    "Infrastructural Impact",
    "Political Impact",
    "Financial Impact",
    "Ecological Impact",
    "Agricultural Impact",
    "Human Health Impact"
]
groupby = ['Date', 'Type', 'ID']

# 350 和 1300 的 ground truth 对应文件
gold_data_files = {
    "350": "./final_query_annotated_350.csv",
    "1300": "./final_query_annotated_1300.csv"
}

# =========== 解析文件名的函数 ===========
def parse_filename(filename):
    """
    从文件名解析出：
      - model_name (模型名称)
      - hist_mod ("historical" / "modern")
      - shot_count ("350" / "1300")
      - shot_type ("oneshot" / "zeroshot")
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")

    shot_type = parts[-1]    # oneshot / zeroshot
    shot_count = parts[-2]   # 350 / 1300
    hist_mod = parts[-3]     # historical / modern
    model_name = "_".join(parts[:-3])  # 其余部分合并当模型名

    return model_name, hist_mod, shot_count, shot_type


# =========== 评估 Accuracy 的函数 ===========
def evaluate_accuracy(data, gold_data, output_file,
                      model_label, hist_mod_label, shot_count_label, shot_type_label):
    """
    - 计算 Accuracy，并附加 `model_label`, `hist_mod_label`, `shot_count_label`, `shot_type_label`
    - 确保 `gold_data` 里的 `impact_columns` 与 `data` 对齐
    """
    data.columns = [x.capitalize() for x in data.columns]
    if 'Health impact' in data.columns:
        data.rename(columns={'Health impact': "Human health impact"}, inplace=True)

    gold_data.columns = [x.capitalize() for x in gold_data.columns]
    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()

    models = data['Model_type'].unique()
    results = []

    for model in models:
        model_data = data[data['Model_type'] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()

        merged = grouped.join(gold_grouped, how='inner', lsuffix='_model', rsuffix='_gold')

        all_correct = (
            merged[[f"{col}_model" for col in impact_columns]].values ==
            merged[[f"{col}_gold"  for col in impact_columns]].values
        ).all(axis=1)

        accuracy = all_correct.sum() / len(all_correct) if len(all_correct) > 0 else 0

        results.append({
            "Model_Type": model,
            "hist_mod": hist_mod_label,
            "shot_count": shot_count_label,
            "shot_type": shot_type_label,
            "Accuracy": round(accuracy, 4)
        })

    df_result = pd.DataFrame(results)
    print("Accuracy结果:\n", df_result)

    if not os.path.isfile(output_file):
        df_result.to_csv(output_file, index=False)
    else:
        df_result.to_csv(output_file, mode='a', header=False, index=False)


# =========== 遍历 ./split 目录下所有 CSV 文件 ===========
def main():
    root_dir = "./split"
    output_dir = "./row-wise-result"
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):
                model_name, hist_mod, shot_count, shot_type = parse_filename(file)

                # 选择正确的 ground truth 数据
                gold_data_file = gold_data_files.get(shot_count, None)
                if not gold_data_file:
                    print(f"未找到匹配的 gold_data 文件: {file}")
                    continue

                gold_data = pd.read_csv(gold_data_file)
                print(f"检查 {gold_data_file} 的列名:", gold_data.columns.tolist())  # 打印列名

                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path} (使用 {gold_data_file} 作为 gold data)")

                data = pd.read_csv(file_path)

                # 生成输出文件名
                output_file_accuracy = os.path.join(
                    output_dir,
                    f"{model_name}_{hist_mod}_{shot_count}_{shot_type}_row-wise_accuracy.csv"
                )

                # 计算 Accuracy
                evaluate_accuracy(
                    data=data,
                    gold_data=gold_data,
                    output_file=output_file_accuracy,
                    model_label=model_name,
                    hist_mod_label=hist_mod,
                    shot_count_label=shot_count,
                    shot_type_label=shot_type
                )

if __name__ == "__main__":
    main()
