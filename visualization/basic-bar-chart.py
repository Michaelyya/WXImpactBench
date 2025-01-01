import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["results_350_oneshot.csv", "results_350_zeroshot.csv", "results_1300_oneshot.csv", "results_1300_zeroshot.csv"]

for file_path in files:
    # 从 CSV 文件读取数据
    df = pd.read_csv(file_path)

    # 对数据按 Metric 分组
    grouped = df.groupby("Metric")

    # 绘制每个 Metric 的柱状图
    for metric, group in grouped:
        # 准备数据
        x_labels = [
            "Infrastructural impact", "Political impact", "Financial impact", 
            "Ecological impact", "Agricultural impact", "Human health impact"
        ]
        x = np.arange(len(x_labels))
        width = 0.15

        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 6))

        # 为每个模型绘制柱状图
        for i, model in enumerate(group["Model_Type"].unique()):
            model_data = group[group["Model_Type"] == model]
            y_values = model_data[x_labels].values.flatten()
            ax.bar(x + i * width, y_values, width, label=model)

        # 设置 y 轴范围
        ax.set_ylim(0, 1)

        # 设置 y 轴刻度
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # 添加图例、标题和标签
        ax.set_xlabel("Type of Impact")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} of {file_path}")
        ax.set_xticks(x + (width * (len(group["Model_Type"].unique()) - 1) / 2))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # 设置图例的位置到右上角，且在图表外
        ax.legend(title="Model Type", loc="upper left", bbox_to_anchor=(1.05, 1))

        # 调整布局以防止图例覆盖
        plt.tight_layout()
        plt.show()
