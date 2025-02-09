import pandas as pd

def calculate_impact_percentage_by_weather_and_period(file_path):
    """
    按天气类型和时间段（historical/modern）统计每个影响类别（impact）的 1 所占的百分比
    :param file_path: CSV 文件路径
    :return: DataFrame，包含每种天气类型在不同时间段的 impact 统计百分比
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保所需列存在
    required_columns = {"Weather", "Type", "Infrastructural Impact", "Political Impact", "Financial Impact", 
                        "Ecological Impact", "Agricultural Impact", "Human Health Impact"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV 文件缺少必要列: {required_columns - set(df.columns)}")

    # 统计每种天气类型在不同时间段（historical/modern）的总行数
    weather_period_counts = df.groupby(["Weather", "Type"]).size().rename("Total Rows")

    # 统计每种天气类型在不同时间段的各 impact 被标记为 1 的数量
    impact_columns = ["Infrastructural Impact", "Political Impact", "Financial Impact", 
                      "Ecological Impact", "Agricultural Impact", "Human Health Impact"]
    
    impact_sums = df.groupby(["Weather", "Type"])[impact_columns].sum()

    # 计算百分比
    impact_percentage = (impact_sums.div(weather_period_counts, axis=0) * 100).reset_index()

    return impact_percentage

if __name__ == "__main__":
    # 你的 CSV 文件路径
    file_path = "final_query_annotated_350.csv"  # 替换为你的文件路径

    try:
        # 计算 impact 百分比
        impact_summary = calculate_impact_percentage_by_weather_and_period(file_path)

        # 输出统计结果
        print(impact_summary)

        # 可选：保存到新 CSV
        output_file = "impact_percentage_by_weather_and_period.csv"
        impact_summary.to_csv(output_file, index=False)
        print(f"统计结果已保存至 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")
