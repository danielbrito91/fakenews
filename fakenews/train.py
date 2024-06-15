import json

import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from fakenews.read_data import read_fake_recogna


def load_model_and_tokenizer(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return model, tokenizer


def compute_metrics(pred):
    acc = accuracy_score(pred.label_ids, pred.predictions.argmax(1))
    clf_report = classification_report(pred.label_ids, pred.predictions.argmax(1), output_dict=True)
    return {"accuracy": acc,
            "clf_report": clf_report}


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

if __name__ == "__main__":


    # Read data
    df = read_fake_recogna().to_pandas()

    # Split data
    train_full, test = train_test_split(df,
                                test_size=0.2,
                                random_state=42,
                                shuffle=True,
                                stratify=df["label"]
                                )

    train, val = train_test_split(train_full,
                                test_size=0.2,
                                random_state=42,
                                shuffle=True,
                                stratify=train_full["label"]
                                )

    ds = datasets.DatasetDict()
    ds["train"] = datasets.Dataset.from_pandas(train.reset_index(drop=True))
    ds["test"] = datasets.Dataset.from_pandas(test.reset_index(drop=True))
    ds["validation"] = datasets.Dataset.from_pandas(val.reset_index(drop=True))

    # Train model
    fake_recogna_trainer = FakeNewsTrainer(ds)
    tokenized_ds = fake_recogna_trainer.tokenize_ds()
    trainer = fake_recogna_trainer.get_trainer(tokenized_ds)

    trainer.train()

    eval_dict = trainer.evaluate(tokenized_ds["test"])

    with open("reports/model_eval/fake_recogna_eval.json", "w") as f:
        json.dump(eval_dict, f)