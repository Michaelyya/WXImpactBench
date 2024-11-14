import pandas as pd

def merge_text_by_date(full_file, broken_file, output_file):
    # Read the CSVs into DataFrames
    full_df = pd.read_csv(full_file, on_bad_lines='skip')
    broken_df = pd.read_csv(broken_file, on_bad_lines='skip')
    
    # Merge DataFrames on 'Date' column, using left join to preserve broken_df structure
    merged_df = broken_df.merge(full_df[['Date', 'Text']], on='Date', how='left', suffixes=('', '_full'))
    
    # Fill missing text in the broken file with the text from the full file
    merged_df['Text'] = merged_df['Text_full']
    
    # Drop the extra column created by merge
    merged_df.drop(columns=['Text_full'], inplace=True)
    
    # Save the updated DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged data saved to {output_file}")


input_list = [
        "./Corrected_OCRs/blizzard_English_historical_corrected.csv",
        "./Corrected_OCRs/cold_English_historical_corrected.csv",
        "./Corrected_OCRs/deluge_English_historical_corrected.csv",
        "./Corrected_OCRs/drought_English_historical_corrected.csv",
        "./Corrected_OCRs/freezing_English_historical_corrected.csv",
        "./Corrected_OCRs/heat_English_modern_corrected.csv",
        "./Corrected_OCRs/heatwave_English_historical_corrected.csv",
        "./Corrected_OCRs/ice_English_modern_corrected.csv",
        "./Corrected_OCRs/storm_English_modern_corrected.csv",
        "./Corrected_OCRs/thunder_English_historical_corrected.csv",
        "./Corrected_OCRs/thunder_English_modern_corrected.csv"
]
input_file = input_list[1]
merge_text_by_date(input_file, 'Allen-15-sample.csv', 'merged_output.csv')
