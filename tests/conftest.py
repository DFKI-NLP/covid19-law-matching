import pytest

from training.preprocessing.datasets_ import (
    LawMatchingDatasets,
    ClaimExtractionDatasets,
)


@pytest.fixture(scope="session")
def law_matching_datasets():
    return LawMatchingDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db"
    )


@pytest.fixture(scope="session")
def claim_extraction_datasets():
    return ClaimExtractionDatasets.load_from_database(
        database="tests/fixtures/database_fixture.db"
    )
