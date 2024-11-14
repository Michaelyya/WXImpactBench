from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from transformers import BartForConditionalGeneration, TrainingArguments, Trainer


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the summarization pipeline with your chosen model

data = {
    "text": [
        "In the new study, researchers used data from the Multi-Country Multi-City (MCC) Collaborative Research Network that included daily deaths and temperatures from 750 locations across 43 countries..."
        # Add other samples as needed
    ],
    "labels": [
        "{'Vulnerabilities': '...'}"  # This should be a summarization or structured output corresponding to the input text.
    ]
}

dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True)

    # Tokenize the labels as well.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=1024, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

trainer.train()