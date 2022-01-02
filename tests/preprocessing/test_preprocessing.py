import pytest
from unittest.mock import MagicMock

import numpy as np
import vcr
from transformers import AutoTokenizer

from training.preprocessing.datasets_ import ClaimExtractionDatasets
from training.preprocessing import Preprocessor


@pytest.fixture(scope="session")
def claim_extraction_sample():
    datasets = ClaimExtractionDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db"
    )
    return datasets.X[0]


@pytest.mark.skip
def test_preprocessor_align_claim_label_works():
    tokenizer = MagicMock()
    preprocessor = Preprocessor(tokenizer, "claim_extraction")

    input_ids = [102, 8862, 818, 7679, 16300, 30881, 1420, 28237, 103, 0, 0]
    offset_mapping = [
        (0, 0),
        (0, 2),
        (2, 3),
        (4, 6),
        (7, 10),
        (10, 11),
        (12, 14),
        (15, 21),
        (0, 0),
    ]
    claim_offsets = [(4, 11), (15, 21)]

    expected_labels = np.array([0, 0, 0, 1, 2, 2, 0, 1, 0, -100, -100])

    labels = preprocessor.align_claim_labels(
        input_ids, offset_mapping, claim_offsets, 0
    )
    assert (labels == expected_labels).all()


@pytest.mark.skip
@vcr.use_cassette("tests/vcr/tokenizer")
def test_preprocessor_end_to_end_claim_extraction(claim_extraction_datasets):
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
    preprocessor = Preprocessor(tokenizer, "claim_extraction")

    dataset = claim_extraction_datasets.X
    input = preprocessor(dataset)


@vcr.use_cassette("tests/vcr/tokenizer")
def test_preprocessor_end_to_end_law_matching(law_matching_datasets):
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
    preprocessor = Preprocessor(tokenizer, "law_matching")

    input = preprocessor(law_matching_datasets.X)

    assert len(law_matching_datasets.train) == 643
    assert len(law_matching_datasets.test) == 161
