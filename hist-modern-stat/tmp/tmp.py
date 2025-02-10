import pandas as pd

def count_impact_with_total_by_weather_and_period(file_path):
    """
    按天气类型和时间段（historical/modern）统计：
    1. 每个影响类别（impact）被标记为 1 的个数
    2. 该天气类型在该时间段出现的总次数（Total Count）
    
    :param file_path: CSV 文件路径
    :return: DataFrame，包含每种天气类型在不同时间段的 impact 统计个数和总行数
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保所需列存在
    required_columns = {"Weather", "Type", "Infrastructural Impact", "Political Impact", "Financial Impact", 
                        "Ecological Impact", "Agricultural Impact", "Human Health Impact"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV 文件缺少必要列: {required_columns - set(df.columns)}")

    # 统计每种天气类型在不同时间段的各 impact 被标记为 1 的个数
    impact_columns = ["Infrastructural Impact", "Political Impact", "Financial Impact", 
                      "Ecological Impact", "Agricultural Impact", "Human Health Impact"]
    
    impact_counts = df.groupby(["Weather", "Type"])[impact_columns].sum()

    # 统计每种天气类型在不同时间段的总行数
    total_counts = df.groupby(["Weather", "Type"]).size().rename("Total Count")

    # 合并统计结果
    result = impact_counts.join(total_counts).reset_index()

    return result

if __name__ == "__main__":
    # 你的 CSV 文件路径
    file_path = "final_query_annotated_350.csv"  # 替换为你的文件路径

    try:
        # 计算 impact 统计个数和总 count
        impact_summary = count_impact_with_total_by_weather_and_period(file_path)

        # 输出统计结果
        print(impact_summary)

        # 可选：保存到新 CSV
        output_file = "impact_count_with_total_by_weather_and_period.csv"
        impact_summary.to_csv(output_file, index=False)
        print(f"统计结果已保存至 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")
