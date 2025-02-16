## Table of Contents <a name="table_of_contents"></a>

- [Table of Contents](#table_of_contents)
- [Data](#data)
- [How to Use](#usage)
  - [Task1: Multi-Label Classification](#task_1)
  - [Task2: Ranking-Based Question-Answering](#task_2)
- [Citation](#citation)

## Data <a name="data"></a>
The data is obtained through collaboration with a proprietary archive institution and covers two temporal periods.
### Multi-Label Classification
We provide the LongCTX dataset in [datasets/LongCTX_Dataset(350).csv](./datasets/LongCTX_Dataset(350).csv) and the MixedCTX data in [datasets/MixedCTX_Dataset(1386).csv](./datasets/MixedCTX_Dataset(1386).csv).

These datasets contain labeled historical records of disruptive weather events and their societal impacts. Each entry includes temporal information, weather type, article text, and human-annotated binary labels for six distinct impact categories, serving as ground truth.

Example of instances
```csv
ID,Date,Time_Period,Weather_Type,Article,Infrastructural Impact,Political Impact,Financial Impact,Ecological Impact,Agricultural Impact,Human Health Impact
0,18800116,historical,Storm, ... On the 22nd another storm arose, and the sea swept the decks, smashing the bulwarks from the bridge aft, destroying the steering gear and carrying overboard a seaman named Anderson. Next day the storm abated and the ship's course was shaped for this port...,1,0,0,0,0,1
```  
- "ID": Unique identifier for each entry.

- "Date": Date of the weather event in `YYYYMMDD` format. 
  
- "Time_Period": Classification of the historical period.

- "Weather_Type": Type of weather event.

- "Article": Text content extracted from historical newspapers describing the event.

- "Impact Columns": Six ground-truth binary labels indicating the impact of the weather event.


### Question-Answering Ranking
We Share the question-answering candidate pool dataset in [datasets/QACandidate_Pool.csv](./datasets/QACandidate_Pool.csv)

The candidate pool is constructed from the LongCTX dataset, with each article generating a query based on its impact labels.

Example of instances
```csv
id,query,correct_passage_index,passage_1,passage_2, ...,passage_100
0, What specific infrastructure and agricultural impact did the British steamer Canopus experience..., 12, p1, p2, ..., p100
```
- "ID": Unique identifier for each entry.

- "Query": Pseudo question generated for question answering.

- "Correct_passage_index": the index number of the ground-truth passage in the 100 passages candidate pool.

- "Passage columns": Candidate pool [1 ground-truth + 99 noise columns] 

## How to use <a name="usage"></a>
Our code repository provides two tasks to evaluate large language models (LLMs) on understanding the impacts of disruptive weather.

## Multi-Label Classification <a name="task_1"></a> 🔍

The multi-label classification task assesses an LLM’s ability to identify disruptive weather impacts in given articles.

To run this task:
1. navigate to the task directory `./Multi-label_Task`.
2. run `pip install -r requirements.txt` to install the required packages.
3. set `model_name` to your desired model.
4. update `your-input.csv` (options are LongCTX and MixedCTX, see [Data](#data)), and specify  `your-output.csv` for saving results.
6. run the evaluation script: `model_eval.py`.

## Question-Answering Ranking <a name="task_2"></a> 🥇

The question-answering ranking task evaluates an LLM’s ability to determine the likelihood that a given article contains the correct answer based on its parametric knowledge.

To run this task:
1. navigate to the task directory `./QA-ranking_Task`.
2. run `pip install -r requirements.txt` to install the required packages.
3. set `model_name` to your desired model.
4. run `model_eval.py`.
5. for data evaluation, change `your-output.json` in `metrics.py` to the saved result from `model_eval.py`.
6. run the metrics evaluation script `metrics.py`.

## Citation <a name="citation"></a>
Please cite:
```python
MICHAEL ANGELA ALLEN...
```
