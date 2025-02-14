import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取CSV文件，假设文件列名与下方一致
df = pd.read_csv('./confusion-matrix/confusion-matrix_1300_oneshot.csv')  # 请把这里的文件名改成你的实际CSV文件名

def shorten_model_name(name: str) -> str:
    short = name.split('/')[-1]
    short = short.replace('-Instruct-v0.1', '')
    short = short.replace('-Instruct', '')
    return short

# 2) 添加新的列，存放简化后的模型名称
df["Short_Model_Name"] = df["Model_Type"].apply(shorten_model_name)

# 2. 获取所有的影响类别，并按照字母顺序排序
impact_categories = df["Impact_Column"].unique()

# 3. 创建 2 行 × 3 列子图（6 个子图），每个子图绘制一个“Impact_Column”
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axs = axs.flatten()  # 将子图对象展平为一维列表，方便在循环中引用

# 4. 使用更淡的 Pastel 配色：TP / FP / TN / FN
colors = ["#AED581", "#FFB9B9", "#90CAF9", "#FFD180"]
metrics = ["TP", "FP", "TN", "FN"]

for i, cat in enumerate(impact_categories):
    ax = axs[i]
    # 从 DataFrame 中筛选出当前 Impact_Column 对应的行
    cat_df = df[df["Impact_Column"] == cat].sort_values(by="Short_Model_Name",ascending=False)

    # 分别取出模型名称和 TP、FP、TN、FN 的值
    models = cat_df["Short_Model_Name"].values
    tp_values = cat_df["TP"].values
    fp_values = cat_df["FP"].values
    tn_values = cat_df["TN"].values
    fn_values = cat_df["FN"].values

    # 计算 x 轴刻度
    x = np.arange(len(models))

    # 依次绘制堆叠柱：TP → FP → TN → FN
    bar_tp = ax.bar(x, tp_values, color=colors[0], label="TP", edgecolor='black')
    bar_fp = ax.bar(x, fp_values, bottom=tp_values, color=colors[1], label="FP", edgecolor='black')
    tp_fp_sum = tp_values + fp_values
    bar_tn = ax.bar(x, tn_values, bottom=tp_fp_sum, color=colors[2], label="TN", edgecolor='black')
    tp_fp_tn_sum = tp_fp_sum + tn_values
    bar_fn = ax.bar(x, fn_values, bottom=tp_fp_tn_sum, color=colors[3], label="FN", edgecolor='black')

    # 设置子图标题（当前 Impact_Column 的名称）
    ax.set_title(cat, fontsize=12)

    # 设置 x 轴刻度标签为模型名称，并倾斜以防文字重叠
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)

# 5. 在图外添加统一图例（避免每个子图重复）
fig.legend(metrics, loc="upper right", bbox_to_anchor=(0.7, 0.58), ncol=1)

# 6. 设置图表总标题并自动调整子图排版
fig.suptitle("Stacked Bar Chart of TP/FP/TN/FN by Model & Impact Column (1386|Oneshot)", fontsize=16)
plt.tight_layout()
plt.show()
