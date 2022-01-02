import json
import os
import statistics
from datetime import datetime
from typing import Union

import numpy as np
from datasets import load_metric

from preprocessing.datasets_ import (
    LawMatchingDatasets,
    ClaimExtractionDatasets,
)


def eval_k_fold(results):
    # TODO: Is this correct? Is there a better way to do this?
    num_results = len(results)
    keys = list(results[0])
    overall = {}
    for key in keys:
        overall[key] = 0.0

    for result in results:
        for key in keys:
            overall[key] += result[key]

    for key in keys:
        overall[key] /= num_results

    for key in keys:
        overall[key + "_stdev"] = statistics.stdev([result[key] for result in results])

    return overall


def compute_metrics_claim_extraction(p):
    metric = load_metric("seqeval", "IOB2")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list = ["O", "B", "I"]

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics_law_matching(eval_pred):
    metric = load_metric("glue", "mrpc")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def report_results(
    task,
    results,
    datasets: Union[LawMatchingDatasets, ClaimExtractionDatasets],
    parameters=None,
):
    path = "/data/experiments/dehio/bachelor_thesis/results"
    run = 0
    date = datetime.today().strftime("%d.%m.%y")
    name_of_run = f"{task}_{date}_{run}"
    while os.path.exists(f"{path}/{name_of_run}"):
        run += 1
        name_of_run = f"{task}_{date}_{run}"
    full_path = f"{path}/{name_of_run}"
    os.mkdir(full_path)
    with open(f"{full_path}/results.txt", "w+") as file:
        if parameters:
            file.write(json.dumps(parameters, indent=2))
        file.write(json.dumps(results, indent=2))
    datasets.save_to_csv(f"{full_path}/dataset.csv")


def num_of_examples_without_claims(dataset):
    return len([e for e in dataset if sum(filter(lambda x: x >= 0, e["labels"])) == 0])
