import os
import pandas as pd
import re

# 创建存放拆分文件的目录
output_dir = "./split"
os.makedirs(output_dir, exist_ok=True)

# 遍历当前目录及所有子目录，获取 CSV 文件列表
csv_files = []
for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))  # 获取完整路径

# 正则匹配：提取 "350_oneshot" 或 "1300_zeroshot" 之类的部分
pattern = re.compile(r"(\d+_(?:oneshot|zeroshot))")

for file_path in csv_files:
    match = pattern.search(file_path)
    if not match:
        continue  # 忽略无法匹配的文件

    file_suffix = match.group(1)  # 提取 "350_oneshot" 或 "1300_zeroshot"

    # 读取 CSV
    df = pd.read_csv(file_path)

    # 检查是否包含 "Type" 列
    if "Type" not in df.columns:
        print(f"Skipping {file_path}: No 'Type' column found.")
        continue

    # 确保 "Model_Type" 存在
    if "Model_Type" not in df.columns:
        print(f"Skipping {file_path}: No 'Model_Type' column found.")
        continue

    # 拆分数据
    historical_df = df[df["Type"] == "historical"]
    modern_df = df[df["Type"] == "modern"]

    # 处理 "Model_Type"，获取模型名（去掉文件名前缀）
    base_model_name = "_".join(os.path.basename(file_path).split("_")[:-2])  # 去掉最后两个部分

    # 保存拆分后的数据
    if not historical_df.empty:
        historical_output_path = os.path.join(output_dir, f"{base_model_name}_historical_{file_suffix}.csv")
        historical_df.to_csv(historical_output_path, index=False)

    if not modern_df.empty:
        modern_output_path = os.path.join(output_dir, f"{base_model_name}_modern_{file_suffix}.csv")
        modern_df.to_csv(modern_output_path, index=False)

print("拆分完成，所有文件保存在 ./split 目录下！")
