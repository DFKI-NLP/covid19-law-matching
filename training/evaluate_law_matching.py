import csv
from random import randint

import torch
from transformers import (
    TrainingArguments,
    IntervalStrategy,
    AutoTokenizer,
)

from preprocessing.datasets_ import LawMatchingDatasets
from train_law_matching_model import train_law_matching_model
from train_law_matching_model import (
    indices_of_wrong_classifications as bert_wrong_indices,
)

from baseline_law_matching import train_baseline
from baseline_law_matching import (
    indices_of_wrong_classifications as baseline_wrong_indices,
)
from preprocessing import Preprocessor

model_checkpoint = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
preprocessor = Preprocessor(tokenizer, "law_matching")

args = TrainingArguments(
    f"/data/experiments/dehio/models/test-law-matching-{randint(0, 100000)}",
    evaluation_strategy=IntervalStrategy.EPOCH,
    learning_rate=0.00001,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    per_gpu_train_batch_size=1,
    per_gpu_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_accumulation_steps=10,
)


def evaluate():
    datasets = LawMatchingDatasets.load_from_csv("law_matching.csv")
    wrong_predictions = []

    for i, (train_set, test_set) in enumerate(datasets.folds):
        trainer, _ = train_law_matching_model(
            train_set, test_set, args, model_checkpoint, preprocessor, tokenizer
        )
        classifier = train_baseline(train_set)

        baseline_indices = baseline_wrong_indices(test_set, classifier)
        with torch.no_grad():
            torch.cuda.empty_cache()
            bert_indices = bert_wrong_indices(test_set, trainer, preprocessor)

        for i in range(len(test_set)):
            if i in baseline_indices or i in bert_indices:
                wrong_predictions.append(
                    (
                        test_set[i][0],
                        test_set[i][1],
                        test_set[i][2],
                        bool(i not in baseline_indices),
                        bool(i not in bert_indices),
                    )
                )

    with open(
        "/data/experiments/dehio/bachelor_thesis/inspection.csv", "w+", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["Claim", "Subsection", "Label", "Baseline", "Bert"])
        for sample in wrong_predictions:
            writer.writerow(sample)
