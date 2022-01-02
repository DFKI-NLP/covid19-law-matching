import jinja2
from random import randint

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    IntervalStrategy,
)
from preprocessing import Preprocessor
from preprocessing.datasets_ import ClaimExtractionDatasets

from utils import (
    eval_k_fold,
    compute_metrics_claim_extraction,
    report_results,
)


def check_token(prediction, label):
    if prediction == 0 and label == 0:
        "Correctly predicted as not being part of a claim"
        return "black"
    if prediction == 1 and label == 1:
        "Correctly predicted as the claim start"
        return "green"
    if prediction == 2 and label == 2:
        "Correctly predicted as being part of the claim"
        return "green"
    if prediction == 1 and label == 2:
        "Correctly predicted as claim, but should be inside of claim"
        return "lightgreen"
    if prediction == 2 and label == 1:
        "Correctly predicted as claim, but should be new of claim"
        return "darkgreen"
    if prediction == 0 and label in [1, 2]:
        "Incorrectly not labeled as claim"
        return "red"
    if prediction in [1, 2] and label == 0:
        "Incorrectly labeled as claim"
        return "blue"
    return "white"


def create_inspection_list(preds, labels, input_ids):
    previous_token = None
    chunks = []
    current_chunk = None
    for i in range(len(input_ids)):
        current_token = check_token(preds[i], labels[i])

        if previous_token is None or current_token != previous_token:
            "New chunk"
            if current_chunk is not None:
                chunks.append(current_chunk)
            current_chunk = (current_token, [input_ids[i]])
        else:
            current_chunk[1].append(input_ids[i])

        previous_token = current_token

    return chunks


def transform_chunk(chunk, tokenizer):
    label, input_ids = chunk
    return (label, tokenizer.decode(input_ids))


def render_html(list_of_samples, results):
    TEMPLATE = """<div> {{ results }}</div>
    {% for sample in list_of_samples -%}<p>
    {% for chunk in sample -%}<span style="color: {{ chunk[0] }};"> {{ chunk[1] }} </span>{% endfor -%}</p><hr>
    <p>{% for chunk in sample -%}\\highlight[green]{ {{ chunk[0] }} }[{{ chunk[1] }}]{% endfor -%}{% endfor -%}</p><hr>
    """
    return (
        jinja2.Environment()
        .from_string(TEMPLATE)
        .render(list_of_samples=list_of_samples, results=results)
    )


def save_results(preds_labels_inputs_ids, results, tokenizer, name):
    list_of_samples = map(
        lambda chunks: map(lambda chunk: transform_chunk(chunk, tokenizer), chunks),
        map(lambda t: create_inspection_list(*t), preds_labels_inputs_ids),
    )
    html = render_html(list_of_samples, results)
    with open(
        f"/data/experiments/dehio/bachelor_thesis/inspection/ce_{name}.html",
        "w+",
    ) as file:
        file.write(html)


def run_experiment3(
    epochs: int = 30,
    learning_rate: float = 2e-5,
):

    datasets = ClaimExtractionDatasets.load_from_database()
    for i, (train_set, test_set) in enumerate(datasets.folds):
        for model_checkpoint, seed in [
            ("deepset/gbert-large", 0),
            ("deepset/gelectra-large", 0),
        ]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            except EnvironmentError:
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepset/gbert-large"
                )  # in case only model weights were saved

            preprocessor = Preprocessor(tokenizer, "claim_extraction")
            results = []

            model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint, num_labels=3, ignore_mismatched_sizes=True
            )
            args = TrainingArguments(
                f"/data/experiments/dehio/models/test-claim-extraction-{randint(0, 100000)}",
                evaluation_strategy=IntervalStrategy.EPOCH,
                learning_rate=learning_rate,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                per_gpu_train_batch_size=1,
                num_train_epochs=epochs,
                weight_decay=0.01,
                seed=seed,
            )
            train_dataset = preprocessor(train_set)
            test_dataset = preprocessor(test_set)
            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_claim_extraction,
            )
            trainer.train()
            result = trainer.evaluate()
            results.append(result)
            output = trainer.predict(test_dataset)
            labels = output[1]
            preds = np.argmax(output[0], axis=2)
            save_results(
                zip(preds, labels, map(lambda x: x["input_ids"], test_dataset)),
                result,
                tokenizer,
                model_checkpoint.split("/")[1] + f"_fold{i}",
            )
            print(f"Results for fold {i}: {result}")
