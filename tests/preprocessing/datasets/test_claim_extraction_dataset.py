import sqlite3

import pytest
import spacy

from training.preprocessing.datasets_ import ClaimExtractionDatasets


@pytest.fixture
def rows_from_database():
    connection = sqlite3.connect("tests/fixtures/database_fixture.db")
    cursor = connection.cursor()
    cursor.execute(
        "SELECT f.url, plaintext, claim FROM fulltext f INNER JOIN claims c on f.url=c.url"
    )
    rows = cursor.fetchall()
    cursor.close()
    return rows


def test_claim_extraction_dataset_should_load_from_database():
    datasets = ClaimExtractionDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db"
    )
    assert datasets.X is not None


def test_claim_extraction_dataset_should_group_rows():
    datasets = ClaimExtractionDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db"
    )
    assert len(datasets.X) == 154
    assert isinstance(datasets.X[0][0], str)  # fulltext
    assert isinstance(datasets.X[0][1], list)  # list of (claim_start, claim_end) tuples
    assert isinstance(datasets.X[0][1][0], tuple)  # (claim_start, claim_end) tuple
    assert isinstance(datasets.X[0][1][1][0], int)  # claim_start, or claim_end


def test_claim_extraction_dataset_folds():
    datasets = ClaimExtractionDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db", folds=5
    )
    for train, test in datasets.folds:
        assert 122 <= len(train) <= 127
        assert 30 <= len(test) <= 34

    assert len(datasets.train) == 134
    assert len(datasets.test) == 24


def test_claim_extraction_datasets_chunking_works(rows_from_database):

    grouped_rows = ClaimExtractionDatasets.group_rows(rows_from_database)
    sample_text, claim_offsets = grouped_rows[0]
    nlp = spacy.load("de_core_news_sm")
    chunks = ClaimExtractionDatasets.chunk_fulltext(sample_text, claim_offsets, nlp)

    assert len(
        [claim for chunk, chunk_claims in chunks for claim in chunk_claims]
    ) == len(
        claim_offsets
    )  # all claims are still here

    for chunk_text, _ in chunks:
        assert len(chunk_text) <= 2550


@pytest.mark.skip
def test_limit_samples_without_claims_works(rows_from_database):
    grouped_rows = ClaimExtractionDatasets.group_rows(rows_from_database)
    sample_text, claim_offsets = grouped_rows[0]
    nlp = spacy.load("de_core_news_sm")
    chunks = ClaimExtractionDatasets.chunk_fulltext(sample_text, claim_offsets, nlp)
    samples = []
    for sample in chunks:
        samples.append(sample)
    ClaimExtractionDatasets.limit_samples_without_claims(samples)
