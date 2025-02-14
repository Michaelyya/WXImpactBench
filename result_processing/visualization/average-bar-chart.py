import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "overall.csv" 
df = pd.read_csv(file_path)

columns = ["350-zero", "350-one", "1300-zero", "1300-one"]
metrics = df["Metric"].unique()
models = df["Model_Type"].unique()

for col in columns:
    x_labels = metrics
    x = np.arange(len(x_labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_data = df[df["Model_Type"] == model]
        y_values = model_data[col].values
        ax.bar(x + i * width, y_values, width, label=model)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Metric")
    ax.set_ylabel("Values")
    ax.set_title(f"Performance Comparison - {col}")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend(title="Model Type", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
