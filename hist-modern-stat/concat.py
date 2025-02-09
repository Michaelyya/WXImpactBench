import os
import pandas as pd

# 设定文件夹路径
folder_path = './result'

# 获取所有csv文件的路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 用一个空的 DataFrame 来存储合并后的数据
merged_df = pd.DataFrame()

# 读取每个CSV文件并连接到merged_df
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    temp_df = pd.read_csv(file_path)  # 读取CSV文件
    merged_df = pd.concat([merged_df, temp_df], ignore_index=True)  # 合并数据

# 输出合并后的数据到一个新的CSV文件
merged_df.to_csv('./merged_result.csv', index=False)

print("合并完成，结果保存在 './merged_result.csv'")
