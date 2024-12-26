import pandas as pd

impact_columns = [
    "Infrastructural impact", 
    "Political impact", 
    "Economic impact", 
    "Ecological impact", 
    "Agricultural impact", 
    "Human health impact"
]
groupby=["Date","Type"]
gold_data = pd.read_csv("/content/long_context_labelled_y.csv")
gold_data.columns = [x.capitalize() for x in gold_data.columns]

def evaluate_accuracy(data, output_file):
    data.columns = [x.capitalize() for x in data.columns]
    models = data["Model_type"].unique()
    gold_grouped = gold_data.groupby(groupby)[impact_columns].max()
    results = []

    for model in models:
        model_data = data[data["Model_type"] == model]
        grouped = model_data.groupby(groupby)[impact_columns].max()

        merged = grouped.join(gold_grouped, how="inner", lsuffix="_model", rsuffix="_gold")
        acc = {col: (merged[f"{col}_model"] == merged[f"{col}_gold"]).sum() / len(merged) for col in impact_columns}
        acc["Model_Type"] = model
        results.append(acc)
    df_result=pd.DataFrame(results)
    print(df_result)
    df_result.to_csv(output_file, index=False)

data = pd.read_csv("/content/output_gpt.csv")
evaluate_accuracy(data, "accuracy_results.csv")