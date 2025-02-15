## Table of Contents <a name="table_of_contents"></a>

- [Table of Contents](#table_of_contents)
- [Data](#data)
- [Current Benchmark Result](#result)
- [How to Use](#usage)
  - [Task1: Multi-Label Classification](#task_1)
  - [Task2: Ranking-Based Question-Answering](#autatic_metrics)
- [Citation](#citation)

## Data <a name="data"></a>
The data is obtained through collaboration with a proprietary archive institution and covers two temporal periods.
### Task 1
We share LongCTX data in [datasets/LongCTX_Dataset(350).csv](./datasets/LongCTX_Dataset(350).csv) and MixedCTX data in [datasets/MixedCTX_Dataset(1386).csv](./datasets/MixedCTX_Dataset(1386).csv)

The dataset consists of labeled historical disruptive weather events and their societal impacts. Each record includes temporal information, weather type, article text, and human-interpreted (ground-truth) binary labels for six different impact categories.

Example of instances
```csv
ID,Date,Time_Period,Weather_Type,Article,Infrastructural Impact,Political Impact,Financial Impact,Ecological Impact,Agricultural Impact,Human Health Impact
0,18800116,historical,Storm, ... On the 22nd another storm arose, and the sea swept the decks, smashing the bulwarks from the bridge aft, destroying the steering gear and carrying overboard a seaman named Anderson. Next day the storm abated and the ship's course was shaped for this port...,1,0,0,0,0,1
```  
- ID: Unique identifier for each entry.

- Date: Date of the weather event in `YYYYMMDD` format. 
  
- Time_Period: Classification of the historical period.

- Weather_Type: Type of weather event.

- Article: Text content extracted from historical newspapers describing the event.

- Impact Columns: Six ground-truth binary labels indicating the impact of the weather event.


### Task 2



## Current Benchmark Result <a name="result"></a>





