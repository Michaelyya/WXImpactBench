import os
import pandas as pd

folder_path = './result'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
merged_df = pd.DataFrame()

# Read and merge all CSV files
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    temp_df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

merged_df.to_csv('./merged_result.csv', index=False)
print("Merging complete. Results saved in './merged_result.csv'")
