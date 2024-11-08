import pandas as pd
import re
import os

def post_ocr_report(original_ocr_file, corrected_ocr_file):
    df_origin = pd.read_csv(original_ocr_file).sort_values(by="Date")
    df_corrected = pd.read_csv(corrected_ocr_file).sort_values(by="Date")
    print(f"Number of duplicated rows in origin file: {df_origin.duplicated(subset='Date').sum()}")
    if df_corrected.duplicated(subset='Date').sum()>0:
        print(f"Number of duplicated rows in corrected file: {df_corrected.duplicated(subset='Date').sum()}")
        print(f"Duplicated rows in corrected file: \n{df_corrected[df_corrected.duplicated(['Date'], keep=False)]}\n-------------------------")
        df_corrected.to_csv(os.path.splitext(os.path.basename(str(corrected_ocr_file)))[0]+"_duplicatedRemoved.csv", index=False)
    else:
        print("No duplicated rows in corrected files!")
    topic = re.search(r"^[^_]+", os.path.basename(str(original_ocr_file))).group().strip()
    
    df_origin = df_origin.drop_duplicates(subset="Date")
    df_corrected = df_corrected.drop_duplicates(subset="Date")
    df_merged = pd.merge(df_origin, df_corrected, on="Date", how="left", suffixes=("_origin", "_corrected"))

    txt_diff_origin_more = 0
    txt_diff_corrected_more = 0
    count1,count2 = 0,0
    origin_topic_count,corrected_topic_count = 0,0

    for index, row in df_merged.iterrows():
        text_origin = str(row["Text_origin"]) if pd.notna(row["Text_origin"]) else ""
        Text_corrected =str(row["Text_corrected"]) if pd.notna(row["Text_corrected"]) else ""
        #accumulate and count the difference in length of texts
        len_diff=len(text_origin)-len(Text_corrected)
        if len_diff > 0: 
            txt_diff_origin_more += len_diff
            count1 += 1
        elif len_diff < 0: 
            txt_diff_corrected_more += abs(len_diff)
            count2 += 1

        #count the difference in frequencies of topic word's appearences
        origin_topic_count+=text_origin.lower().count(topic)
        corrected_topic_count +=Text_corrected.lower().count(topic)

        df_merged.at[index, "length_diff"]=len_diff

    avg_origin_more = txt_diff_origin_more/count1 if count1 > 0 else 0
    avg_corrected_more = txt_diff_corrected_more/count2 if count2 > 0 else 0

    #check texts that we missed or didnt process
    check=(df_merged["Text_origin"]!="[]") & ((df_merged["Text_corrected"].isna()) | (df_merged["Text_corrected"] == "[]"))
    df_result = df_merged[check]
    if df_result.empty:
        print("All texts in origin file are processed!")
    else:
        print("Rows where corrected file is empty but original file has texts.")
        print(df_result[["Date","Text_origin","Text_corrected"]])
        df_result=df_result.rename(columns={"Text_origin":"Text"})
        df_result[["Date", "Text"]].to_csv(os.path.splitext(os.path.basename(str(corrected_ocr_file)))[0]+"_unprocessed.csv", index=False)
    print("-------------------------")

    print(f"Avg text length difference where origin file has longer text: {avg_origin_more:.2f}")
    print(f"Avg text length difference where corrected file has longer text: {avg_corrected_more:.2f}")
    print("-------------------------")
    print(f"topic '{topic}' has number of appearances = {origin_topic_count} times in origin file and {corrected_topic_count} times in corrected file")


if __name__ == "__main__":
    post_ocr_report("../snow_English_historical_ML_corpus.csv", "./snow_english_historical_corrected.csv")
