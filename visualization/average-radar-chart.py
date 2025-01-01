import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file again after confirmation
file_path = 'overall.csv'
df = pd.read_csv(file_path)

# Unique metrics and models
metrics = df['Metric'].unique()
models = df['Model_Type'].unique()
columns = df.columns[2:]  # Exclude 'Model_Type' and 'Metric'

# Generate radar plots
for column in columns:
    # Filter data for each model and normalize
    radar_data = []
    for model in models:
        model_data = df[df['Model_Type'] == model].set_index('Metric')[column]
        radar_data.append(model_data.values)

    # Radar plot setup
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Add each model's data to the radar plot
    for model_data, model in zip(radar_data, models):
        values = list(model_data) + [model_data[0]]  # Close the plot
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    # Customize the radar chart
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, weight="bold", ha='center')

    # Move text labels outside the chart
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = np.cos(angle), np.sin(angle)
        if angle < np.pi / 2 or angle > 3 * np.pi / 2:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
        label.set_verticalalignment("center")
    
    ax.set_title(f"Radar Plot for {column}", size=15, weight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Show the plot
    plt.tight_layout()
    plt.show()