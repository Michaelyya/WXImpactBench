import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 如果想画Heatmap需要导入seaborn
import re

data = [["350_oneshot", "350|Oneshot"],["350_zeroshot","350|Zeroshot"],["1300_oneshot","1386|Oneshot"],["1300_oneshot","1386|Zeroshot"]]
for filename,title in data:
    # 1. 读取 CSV 文件
    df = pd.read_csv(f"./f1_recall_precision_impact-acc/results_{filename}.csv")


    #############################
    # 2. 将宽表转换为长表 (melt)
    #############################
    df_melt = df.melt(
        id_vars=["Model_Type", "Metric"],  # 保留列
        var_name="Impact",                 # 原本impact列名放到 "Impact"
        value_name="Score"                 # 对应数值放到 "Score"
    )
    # 现在 df_melt 包含: [Model_Type, Metric, Impact, Score]

    ##################################
    # 3. 定义对字符串做清理的函数 #
    ##################################
    def shorten_model_name(name: str) -> str:
        """
        去除斜杠前缀 和 -Instruct等后缀, 例如:
        'mistralai/Mixtral-8x7B-Instruct-v0.1' -> 'Mixtral-8x7B'
        """
        # (a) 去掉斜杠前的内容
        short = name.split('/')[-1]
        # (b) 用正则去除 -Instruct* 或 -instruct* 以及其后续字符串
        short = re.sub(r'-[Ii]nstruct.*', '', short)
        return short

    def shorten_impact_name(impact: str) -> str:
        """
        去除单词 " impact"，如 "Infrastructural impact" -> "Infrastructural"
        """
        return impact.replace(" impact", "").strip()

    ###############################################
    # 4. 需要绘制的指标列表 (Precision / Recall / F1 / Accuracy)
    ###############################################
    metrics_list = ["Precision", "Recall", "F1", "Accuracy"] 

    ###############################################################
    # 5. 循环对每个 Metric 生成一个单独的 Heatmap (四张图)
    ###############################################################
    for metric in metrics_list:
        # 5.1 筛选出当前 metric 的行
        subset = df_melt[df_melt["Metric"] == metric]

        # 5.2 pivot： 行->Model_Type, 列->Impact, 值->Score
        pivot_df = subset.pivot(index="Model_Type", columns="Impact", values="Score")

        # 5.3 清理列名（Impact 名）: 去掉 " impact"
        pivot_df = pivot_df.rename(columns=shorten_impact_name)

        # 5.4 清理行名（Model_Type）
        # 方法: 把索引 apply shorten_model_name
        pivot_df.index = pivot_df.index.map(shorten_model_name)

        # 5.5 绘图: 每个 metric 都在自己的 figure 中
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            pivot_df, 
            annot=True,      # 在格子中显示数值
            fmt=".3f",       # 保留 3 位小数
            cmap="YlGnBu"    # 可换 "Blues", "coolwarm" 等
        )

        # 5.6 美化与标题
        plt.title(f"{metric} Heatmap ({title})", fontsize=14)
        plt.xlabel("Impact", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.tight_layout()

        # 5.7 显示 (每次循环都会弹出一个新图)
        plt.show()