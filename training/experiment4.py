import csv
import json
import os
from datetime import datetime
from random import randint

import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    IntervalStrategy,
    AutoModelForSequenceClassification,
    set_seed,
)
from sklearn.metrics import precision_score, recall_score

from preprocessing import Preprocessor
from preprocessing.datasets_ import LawMatchingDatasets

from utils import eval_k_fold, compute_metrics_law_matching


def train_law_matching_model(
    train_set, test_set, args, model_checkpoint, preprocessor, tokenizer
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )
    model.config.gradient_checkpointing = True
    train_dataset = preprocessor(train_set)
    test_dataset = preprocessor(test_set)
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_law_matching,
    )
    trainer.train()
    return trainer, model


def get_classifications(test_dataset, trainer, preprocessor):
    results = []
    processed_test_dataset = preprocessor(test_dataset)
    logits, labels, _ = trainer.predict(processed_test_dataset)
    predictions = np.argmax(logits, axis=1)

    for i in range(len(predictions)):
        results.append(
            (test_dataset[i][0], test_dataset[i][1], test_dataset[i][2], predictions[i])
        )

    return results


date = datetime.today().strftime("%d.%m.%y")
PATH = f"/data/experiments/dehio/bachelor_thesis/results/experiment4_{date}"


def run_experiment4(
    epochs: int = 3,
    learning_rate: float = 0.00001,
):
    set_seed(0)
    datasets = LawMatchingDatasets.load_from_csv("data/law_matching.csv")

    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    results = {
        "deepset/gbert-large": [],
        "deepset/gbert-base": [],
        "deepset/gelectra-large": [],
        "deepset/gelectra-base": [],
    }
    predictions = {}

    for i, (train_set, test_set) in enumerate(datasets.folds):
        for model_checkpoint in results.keys():
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            preprocessor = Preprocessor(tokenizer, "law_matching")
            args = TrainingArguments(
                f"/data/experiments/dehio/models/test-law-matching-{randint(0, 100000)}",
                evaluation_strategy=IntervalStrategy.EPOCH,
                learning_rate=learning_rate,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                per_gpu_train_batch_size=1,
                num_train_epochs=epochs,
                weight_decay=0.01,
                seed=100,
            )
            trainer, model = train_law_matching_model(
                train_set, test_set, args, model_checkpoint, preprocessor, tokenizer
            )
            preds, label_ids, result = trainer.predict(test_set)
            result["precision"] = precision_score(preds, label_ids)
            result["recall"] = recall_score(preds, label_ids)

            if model_checkpoint not in results:
                results[model_checkpoint] = []
            results[model_checkpoint].append(result)

            with torch.no_grad():
                torch.cuda.empty_cache()
                classifications = get_classifications(test_set, trainer, preprocessor)

                for classification in classifications:
                    if (classification[0], classification[1]) not in predictions:
                        predictions[(classification[0], classification[1])] = {
                            "label": classification[2],
                            model_checkpoint: classification[3],
                        }
                    else:
                        predictions[(classification[0], classification[1])][
                            model_checkpoint
                        ] = classification[3]

            torch.cuda.empty_cache()
            del trainer
            del model
            print(f"Results for fold {i}: {result}")
            del result

    with open(f"{PATH}/results.txt", "w+") as file:
        for model_checkpoint, result_list in results.items():
            file.write(
                json.dumps(
                    {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "model": model_checkpoint,
                    },
                    indent=2,
                )
            )
            file.write("\nResult over all folds")
            file.write(json.dumps(eval_k_fold(result_list), indent=2))
    with open(f"{PATH}/inspection.csv", "w+") as file:
        fields = [
            {"Claim": key[0], "Subsection": key[1], **value}
            for key, value in predictions.items()
        ]
        fieldnames = [
            "Claim",
            "Subsection",
            "label",
            "deepset/gbert-base",
            "deepset/gbert-large",
            "deepset/gelectra-base",
            "deepset/gelectra-large",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for field in fields:
            writer.writerow(field)
    datasets.save_to_csv(f"{PATH}/dataset.csv")
