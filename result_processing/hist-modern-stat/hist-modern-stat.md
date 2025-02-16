## Functionality
This part is used to generate a slipped version of the multi-labelling task results by model. 
- You may run the split.py file first to split the original csv
- Run eval.py to get the Precision, Recall, F1, Accuracy
- Run concat.py to concat the eval csvs from the previous step into one csv

- If you want to focus on F1 only. You can run avgf1_eval.py
- If you want to calculate things on row-wisely. You can run row-wise.py, which if one type of impact is incorrect, it will not be considered as correct. 

## **Model Directories**
These directories contain evaluation results or outputs related to different AI models:
- `deepseek-v3` - Files related to the DeepSeek-V3 model.
- `gemma-2-9b` - Files related to the Gemma-2 9B model.
- `gpt-3.5-turbo` - Files related to OpenAI's GPT-3.5 Turbo model.
- `gpt-4` - Files related to OpenAI's GPT-4 model.
- `gpt-4o` - Files related to OpenAI's GPT-4o model.
- `Llama-3.1-8B-Instruct` - Files related to the Llama 3.1 8B Instruct model.
- `Mistral-Small-24B` - Files related to the Mistral Small 24B model.
- `mixtral-7B` - Files related to the Mixtral 7B model.
- `Mixtral-8x7B` - Files related to the Mixtral 8×7B model.
- `Qwen-2.5-7B` - Files related to the Qwen 2.5 7B model.
- `Qwen-2.5-14B` - Files related to the Qwen 2.5 14B model.

## **Evaluation Result Directories**
These directories store results from different evaluation metrics:
- `f1avg_result` - Contains Micro-Averaged F1 Score evaluation results.
- `row-wise-result` - Contains row-wise accuracy evaluation results.
- `result` - General results directory (purpose unclear, may store aggregated metrics).

- `split` - Stores the processed CSV files that are split into different segments for evaluation.