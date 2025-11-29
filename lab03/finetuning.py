import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

dataset = load_dataset("jziebura/polish_youth_slang_classification")
dataset = dataset.rename_column("sentyment", "labels")

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-cased", num_labels=3)


def tokenize(examples):
    return tokenizer(examples["tekst"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, average="weighted")}


dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="herbert-slang",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # type: ignore
    eval_dataset=dataset["validation"],  # type: ignore
    compute_metrics=compute_metrics,
).train()
