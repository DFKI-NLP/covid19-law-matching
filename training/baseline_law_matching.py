import string
from typing import Optional

import spacy
from datasets import load_metric
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

from preprocessing.datasets_ import LawMatchingDatasets
from utils import report_results, eval_k_fold

nlp = spacy.load("de_core_news_sm")


def _tokenizer(text):
    doc = nlp(text)
    return [
        token.text
        for token in doc
        if not token.is_stop and not (token.text in string.punctuation)
    ]


classifier = LogisticRegression()
vectorizer = TfidfVectorizer(tokenizer=_tokenizer, preprocessor=None)
metric = load_metric("glue", "mrpc")


def preprocess(X):
    texts = []
    x_set = []
    y_set = []
    for claim, subsection, label in X:
        texts.append(claim)
        texts.append(subsection)
        y_set.append(int(label))
    vectors = vectorizer.fit_transform(texts)
    for i in range(0, vectors.shape[0], 2):
        x_set.append(cosine_similarity(vectors[i], vectors[i + 1])[0])
    return x_set, y_set


def calculate_baseline_law_matching(from_file: Optional[str]):
    datasets = LawMatchingDatasets.load_from_csv(from_file)
    results = []

    for train_set, test_set in datasets.folds:
        x_train, y_train = preprocess(train_set)
        x_test, y_test = preprocess(test_set)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        result = metric.compute(predictions=predictions, references=y_test)
        result["precision"] = precision_score(y_test, predictions)
        result["recall"] = recall_score(y_test, predictions)
        results.append(result)

    print(f"Overall results: {eval_k_fold(results)}")
    report_results("baseline", eval_k_fold(results), datasets)


def train_baseline(train_dataset):
    x_train, y_train = preprocess(train_dataset)
    classifier.fit(x_train, y_train)

    return classifier


def indices_of_wrong_classifications(test_dataset, classifier):
    x_test, y_test = preprocess(test_dataset)
    predictions = classifier.predict(x_test)
    wrong_predictions = []

    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            wrong_predictions.append(i)

    result = metric.compute(predictions=predictions, references=y_test)
    print(
        "Results for baseline:",
        result,
        f"{(len(x_test) - len(wrong_predictions)) /len(x_test)}",
    )
    return wrong_predictions
