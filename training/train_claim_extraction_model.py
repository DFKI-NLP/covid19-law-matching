from random import randint

import numpy as np
from datasets import load_metric
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
    num_of_examples_without_claims,
    report_results,
)


def train_claim_extraction(
    epochs: int = 3,
    cross_validation: bool = True,
    inspect: bool = False,
    learning_rate: float = 2e-5,
    model_checkpoint: str = "deepset/gbert-large",
):
    args = TrainingArguments(
        f"/data/experiments/dehio/models/test-claim-extraction-{randint(0, 100000)}",
        evaluation_strategy=IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        per_gpu_train_batch_size=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )
    model_name = model_checkpoint.split("/")[-1]

    datasets = ClaimExtractionDatasets.load_from_database()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    except EnvironmentError:
        tokenizer = AutoTokenizer.from_pretrained(
            "deepset/gbert-large"
        )  # in case only model weights were saved

    preprocessor = Preprocessor(tokenizer, "claim_extraction")
    results = []

    if cross_validation:
        for i, (train_set, test_set) in enumerate(datasets.folds):
            model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint, num_labels=3, ignore_mismatched_sizes=True
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

            print(f"Results for fold {i}: {result}")

        print(f"Overall results: {eval_k_fold(results)}")
        report_results(
            "claim_extraction",
            eval_k_fold(results),
            datasets,
            parameters={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "model": model_checkpoint,
            },
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, num_labels=3, ignore_mismatched_sizes=True
        )

        train_dataset = preprocessor(datasets.train)
        test_dataset = preprocessor(datasets.test)

        print(
            "Examples with no claims in train dataset:",
            num_of_examples_without_claims(train_dataset),
        )
        print(
            "Examples with no claims in test dataset:",
            num_of_examples_without_claims(test_dataset),
        )
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_claim_extraction,
        )
        trainer.train()
        if inspect:
            metric = load_metric("seqeval", "IOB2")

            def inspect_sample(nr: int):
                output = model(
                    input_ids=test_dataset[nr]["input_ids"].unsqueeze(0).cuda(),
                    attention_mask=test_dataset[nr]["attention_mask"]
                    .unsqueeze(0)
                    .cuda(),
                )
                logits = output.logits.cpu().detach().numpy()
                pred = np.argmax(logits, axis=2)
                original_text_raw = (
                    test_dataset[nr]["input_ids"].detach().numpy().copy()
                )
                text_raw = original_text_raw.copy()
                original_text_raw[np.array(test_dataset[nr]["labels"]) == 0] = 0
                text_raw[pred[0] == 0] = 0
                print("Target text:\n")
                print(tokenizer.decode(original_text_raw))
                print("Inferred text:")
                print(tokenizer.decode(text_raw))
                print("Predictions:")
                print(pred)
                label_list = ["O", "B", "I"]
                true_predictions = [
                    label_list[p]
                    for (p, l) in zip(pred, test_dataset[nr]["labels"])
                    if l != -100
                ]
                true_labels = [
                    label_list[l]
                    for (p, l) in zip(pred, test_dataset[nr]["labels"])
                    if l != -100
                ]
                print(
                    metric.compute(predictions=true_predictions, references=true_labels)
                )

            breakpoint()
        result = trainer.evaluate()
        print(f"Results: {result}")

        parameter = {"epochs": epochs, "learning_rate": learning_rate}
