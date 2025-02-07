import pandas as pd
import matplotlib.pyplot as plt

data = [["350_oneshot", "350|Oneshot"], ["350_zeroshot", "350|Zeroshot"], ["1300_oneshot", "1386|Oneshot"], ["1300_zeroshot", "1386|Zeroshot"]]

for filename, titlename in data:
    # 假设CSV文件名为 'accuracy.csv'
    df = pd.read_csv(f'./row-wise-accuracy/row-wise-accuracy_{filename}.csv')

    # 定义函数：去除路径、-Instruct等后缀
    def shorten_model_name(name: str) -> str:
        short = name.split('/')[-1]       # 去掉斜杠前的部分
        short = short.replace('-Instruct-v0.1', '')
        short = short.replace('-Instruct', '')
        return short


    df['Short_Model_Name'] = df['Model_Type'].apply(shorten_model_name)

    # 按 Accuracy 从高到低排序
    df_sorted = df.sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(8, 4))

    # 绘制柱状图并保存 bar 对象
    bars = plt.bar(df_sorted['Short_Model_Name'], df_sorted['Accuracy'],
                color='skyblue', edgecolor='black')

    plt.title(f'Row-Wise Accuracy by Model ({titlename})', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Row-Wise Accuracy', fontsize=12)

    # x轴标签倾斜，以免重叠
    plt.xticks(rotation=45, ha='right')

    # 将y轴范围限制为 [0, 0.5]
    plt.ylim([0, 0.5])

    # 在每个柱子上方显示准确率数值
    for bar in bars:
        height = bar.get_height()  # 当前柱子的高度
        # x 坐标置于柱子中央, y 坐标略高于柱子的顶部, 显示到小数点后4位
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # 柱子中心
            height + 0.005,                    # 比柱子顶部略高一点
            f"{height:.4f}",                   # 4位小数
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    plt.show()