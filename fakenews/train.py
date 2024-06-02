import datasets
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_model_and_tokenizer(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return model, tokenizer


def compute_metrics(pred):
    acc = accuracy_score(pred.label_ids, pred.predictions.argmax(1))
    return {"accuracy": acc}


class FakeNewsTrainer:
    def __init__(
        self,
        ds: datasets.dataset_dict.DatasetDict,
        model_name: str = "adalbertojunior/distilbert-portuguese-cased",
    ):
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.ds = ds

    def tokenize_function(self, example):
        return self.tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=512
        )

    def tokenize_ds(self) -> datasets.dataset_dict.DatasetDict:
        tokenized_ds = self.ds.map(self.tokenize_function)
        tokenized_ds = tokenized_ds.remove_columns(["text"])
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch")
        return tokenized_ds

    def get_trainer(self, tokenized_ds: datasets.dataset_dict.DatasetDict):
        train_args = TrainingArguments(
            learning_rate=2e-5,
            push_to_hub=False,
            output_dir="output",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        return Trainer(
            self.model,
            train_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
