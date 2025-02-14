import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["results_350_oneshot.csv", "results_350_zeroshot.csv", "results_1300_oneshot.csv", "results_1300_zeroshot.csv"]

for file_path in files:
    df = pd.read_csv(file_path)
    grouped = df.groupby("Metric")
    for metric, group in grouped:

        x_labels = [
            "Infrastructural impact", "Political impact", "Financial impact", 
            "Ecological impact", "Agricultural impact", "Human health impact"
        ]
        x = np.arange(len(x_labels))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model in enumerate(group["Model_Type"].unique()):
            model_data = group[group["Model_Type"] == model]
            y_values = model_data[x_labels].values.flatten()
            ax.bar(x + i * width, y_values, width, label=model)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        ax.set_xlabel("Type of Impact")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} of {file_path}")
        ax.set_xticks(x + (width * (len(group["Model_Type"].unique()) - 1) / 2))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        ax.legend(title="Model Type", loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
