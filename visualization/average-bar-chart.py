import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从 overall.csv 文件读取数据
file_path = "overall.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 提取列名
columns = ["350-zero", "350-one", "1300-zero", "1300-one"]
metrics = df["Metric"].unique()
models = df["Model_Type"].unique()

# 绘制柱状图
for col in columns:
    # 准备数据
    x_labels = metrics
    x = np.arange(len(x_labels))
    width = 0.15

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        # 获取当前模型的每个指标的数据
        model_data = df[df["Model_Type"] == model]
        y_values = model_data[col].values

        # 绘制每个模型的数据
        ax.bar(x + i * width, y_values, width, label=model)

    # 设置 y 轴范围
    ax.set_ylim(0, 1)

    # 设置 y 轴刻度
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # 添加图例、标题和标签
    ax.set_xlabel("Metric")
    ax.set_ylabel("Values")
    ax.set_title(f"Performance Comparison - {col}")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend(title="Model Type", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
