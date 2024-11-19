import pandas as pd

# Define core and general keywords
core_keywords = ["blizzard", "cold", "deluge", "drought", "flood", "freezing", "heat", "heatwave", "ice", "rain", "snow", "snowstorm", "storm", "thunder", "torrential"]
general_keywords = ["crisis", "rescue", "safety", "hazard", "risk", "catastrophe", "casualties", "destruction", "resilience", "adaptation", "death"]
temporal_trends = 'Y'


# temporal_trends = 'M'


# Load the CSV file
def load_data(filepath):
    return pd.read_csv(filepath)


# 1. Number of Rows
def get_number_of_rows(df):
    return len(df)


# 2. Time Gaps
def get_time_gaps(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    # Daily gaps
    df_daily = df.set_index('Date').resample('D').count()
    day_gaps = df_daily[df_daily['Text'] == 0]

    # Monthly gaps
    df_monthly = df.set_index('Date').resample('ME').count()
    month_gaps = df_monthly[df_monthly['Text'] == 0]

    # Yearly gaps
    df_yearly = df.set_index('Date').resample('YE').count()
    year_gaps = df_yearly[df_yearly['Text'] == 0]

    return len(day_gaps), len(month_gaps), len(year_gaps)


# 3. Keyword Frequency
def get_keyword_frequency(df):
    keyword_counts = {keyword: 0 for keyword in core_keywords + general_keywords}
    for text in df['Text']:
        if isinstance(text, str):
            for keyword in keyword_counts.keys():
                if keyword in text.lower():
                    keyword_counts[keyword] += 1
    return keyword_counts


# 4. Length of Text
def get_average_length_and_word_count(df):
    lengths = df['Text'].str.len()
    word_counts = df['Text'].str.split().str.len()  # Split by whitespace and count the words

    # Calculate statistics for lengths
    length_mean = lengths.mean()
    length_median = lengths.median()
    length_mode = lengths.mode()[0] if not lengths.mode().empty else None
    length_std = lengths.std()

    # Calculate statistics for word counts
    word_count_mean = word_counts.mean()
    word_count_median = word_counts.median()
    word_count_mode = word_counts.mode()[0] if not word_counts.mode().empty else None
    word_count_std = word_counts.std()

    return {
        'length_stats': {
            'mean': length_mean,
            'median': length_median,
            'mode': length_mode,
            'std_dev': length_std
        },
        'word_count_stats': {
            'mean': word_count_mean,
            'median': word_count_median,
            'mode': word_count_mode,
            'std_dev': word_count_std
        }
    }


# 5. Descriptive Statistics
def get_descriptive_statistics(df):
    # Get year
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['year'] = df['Date'].dt.year

    # Create a new column for lengths (in characters)
    df['length'] = df['Text'].str.len()
    # Create a new column for word counts
    df['word_count'] = df['Text'].str.split().str.len()

    # Group by year and calculate statistics for length
    length_statistics = df.groupby('year')['length'].agg(
        mean='mean',
        median='median',
        std_dev=lambda x: x.std() if len(x) > 1 else 0,
        mode=lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index()

    # Group by year and calculate statistics for word count
    word_count_statistics = df.groupby('year')['word_count'].agg(
        mean='mean',
        median='median',
        std_dev=lambda x: x.std() if len(x) > 1 else 0,
        mode=lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index()

    # Combine both statistics into a single result
    result = {}
    for _, length_row in length_statistics.iterrows():
        year = length_row['year']
        word_row = word_count_statistics[word_count_statistics['year'] == year].iloc[0]
        result[year] = {
            'length_mean': length_row['mean'],
            'length_median': length_row['median'],
            'length_mode': length_row['mode'],
            'length_std_dev': length_row['std_dev'],
            'word_count_mean': word_row['mean'],
            'word_count_median': word_row['median'],
            'word_count_mode': word_row['mode'],
            'word_count_std_dev': word_row['std_dev']
        }

    return result


# 6. Temporal Trends
def get_temporal_trends(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    trends = {}
    for keyword in core_keywords + general_keywords:
        trends[keyword] = \
        df[df['Text'].str.contains(keyword, case=False, na=False)].groupby(df['Date'].dt.to_period(temporal_trends)).count()[
            'Text']
    return trends


# Main function to execute the metadata extraction and save to CSV
def extract_metadata(filepath, output_filepath):
    df = load_data(filepath)

    metadata = {
        "Number of Rows": get_number_of_rows(df),
        "Time Gaps": get_time_gaps(df),
        "Keyword Frequency": get_keyword_frequency(df),
        "Average Length": get_average_length_and_word_count(df),
        "Descriptive Statistics": get_descriptive_statistics(df),
        "Temporal Trends": get_temporal_trends(df)
    }

    # Prepare the results for saving in the specified format
    results = []

    # Number of Rows
    results.append(["Number of Rows", metadata["Number of Rows"]] + [""])
    results.append([""])  # Empty row

    # Time Gaps
    day_gaps, month_gaps, year_gaps = metadata["Time Gaps"]
    results.append(["Time Gaps (Day)", day_gaps] + [""])
    results.append(["Time Gaps (Month)", month_gaps] + [""])
    results.append(["Time Gaps (Year)", year_gaps] + [""])
    results.append([""])  # Empty row

    # Text Length
    results.append(["Text length"] + [""])
    avg_length = metadata["Average Length"]
    results.append(
        ["Char Mean", "Char Median", "Char Mode", "Char Std Dev", "", "Word Mean", "Word Median", "Word Mode",
         "Word Std Dev"]
    )
    results.append(
        [avg_length['length_stats']['mean'], avg_length['length_stats']['median'], avg_length['length_stats']['mode'],
         avg_length['length_stats']['std_dev']] +
        [""] +
        [avg_length['word_count_stats']['mean'], avg_length['word_count_stats']['median'],
         avg_length['word_count_stats']['mode'], avg_length['word_count_stats']['std_dev']]
    )
    results.append([""])  # Empty row

    # Descriptive Statistics
    results.append(["Descriptive Statistics (Year)", "Char Mean", "Char Median", "Char Mode", "Char Std Dev"] +
                   [""] + ["Word Mean", "Word Median", "Word Mode", "Word Std Dev"])
    statistics = metadata["Descriptive Statistics"]
    # Loop through each year and append its statistics
    for year, stats in statistics.items():
        results.append(
            [year, stats['length_mean'], stats['length_median'], stats['length_mode'], stats['length_std_dev']] + [""] +
            [stats['word_count_mean'], stats['word_count_median'], stats['word_count_mode'],
             stats['word_count_std_dev']] + [""])
    results.append([""])  # Empty row for spacing

    # Keyword Frequency
    results.append(["Keyword Frequency"] + core_keywords + general_keywords)
    results.append(
        ["All"] + [metadata["Keyword Frequency"].get(keyword, 0) for keyword in core_keywords + general_keywords])
    results.append([""])  # Empty row

    # Distribution of Core Keywords
    results.append(["Temporal Trends"] + core_keywords + general_keywords)
    trends_df = pd.DataFrame(metadata["Temporal Trends"]).fillna(0)
    for month in trends_df.index:
        month_str = month.strftime("%Y%m") if temporal_trends == 'M' else month.strftime("%Y")
        results.append([month_str] + trends_df.loc[month].tolist() + [""])

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filepath, index=False, header=False)


input_list = [
    "./All-Corrected-OCRs/blizzard_English_historical_corrected.csv",
    "./All-Corrected-OCRs/blizzard_English_modern_corrected.csv",
    "./All-Corrected-OCRs/cold _English_modern_ML_corpus_corrected.csv",
    "./All-Corrected-OCRs/cold_English_historical_corrected.csv",
    "./All-Corrected-OCRs/deluge_English_historical_corrected.csv",
    "./All-Corrected-OCRs/deluge_English_modern_ML_corpus_corrected.csv",
    "./All-Corrected-OCRs/drought_English_historical_corrected.csv",
    "./All-Corrected-OCRs/flood_english_historical_corrected.csv",
    "./All-Corrected-OCRs/freezing_English_historical_corrected.csv",
    "./All-Corrected-OCRs/freezing_English_modern_corrected.csv",
    "./All-Corrected-OCRs/heat_English_modern_corrected.csv",
    "./All-Corrected-OCRs/heatwave_English_historical_corrected.csv",
    "./All-Corrected-OCRs/heatwave_English_modern_corrected.csv",
    "./All-Corrected-OCRs/ice_english_historical_corrected.csv",
    "./All-Corrected-OCRs/ice_English_modern_corrected.csv",
    "./All-Corrected-OCRs/rain_english_historical_corrected.csv",
    "./All-Corrected-OCRs/rain_English_modern_corrected.csv",
    "./All-Corrected-OCRs/snow_english_historical_corrected.csv",
    "./All-Corrected-OCRs/snowstorm_English_modern_corrected.csv",
    "./All-Corrected-OCRs/storm_english_historical_corrected.csv",
    "./All-Corrected-OCRs/storm_English_modern_corrected.csv",
    "./All-Corrected-OCRs/thunder_English_historical_corrected.csv",
    "./All-Corrected-OCRs/thunder_English_modern_corrected.csv",
    "./All-Corrected-OCRs/torrential_english_historical_corrected.csv"
]

type_list = [
    "blizzard_old",
    "blizzard_modern",
    "cold_modern",
    "cold_old",
    "deluge_old",
    "deluge_modern",
    "drought_old",
    "flood_old",
    "freezing_old",
    "freezing_modern",
    "heat_modern",
    "heatwave_old",
    "heatwave_modern",
    "ice_old",
    "ice_modern",
    "rain_old",
    "rain_modern",
    "snow_old",
    "snowstorm_modern",
    "storm_old",
    "storm_modern",
    "thunder_old",
    "thunder_modern",
    "torrential_old"
]
# extract_metadata("../statistics/FinalQuerySelected.csv", f'FinalQuerySelected_metadata.csv')
# 24 in total?
# i = 3
# i = 22

# i = 23 finished
for i in range(0, len(input_list)):
    if i == 3:
        continue
    print(input_list[i])
    extract_metadata(input_list[i], f'./metadata/{type_list[i]}_metadata.csv')
