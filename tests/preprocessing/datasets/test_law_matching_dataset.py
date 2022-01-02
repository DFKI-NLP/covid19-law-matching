import datetime
import tempfile

from training.preprocessing.datasets_.models import Reference
from training.preprocessing.datasets_ import (
    resolve_reference_to_subsection_text,
    LawMatchingDatasets,
)


def test_law_matching_dataset_should_load_from_database(law_matching_datasets):

    assert law_matching_datasets.X is not None


def test_law_matching_dataset_should_parse_rows(law_matching_datasets):

    assert len(law_matching_datasets.X) == 804  # both positive and negative samples
    assert isinstance(law_matching_datasets.X[0][0], str)  # claim
    assert isinstance(law_matching_datasets.X[0][1], str)  # list of references
    assert isinstance(law_matching_datasets.X[0][2], bool)  # label


def test_law_matching_dataset_should_load_legislation(law_matching_datasets):

    assert law_matching_datasets.acts


def test_law_matching_dataset_is_balanced(law_matching_datasets):

    positive_samples = filter(lambda sample: sample[2] is True, law_matching_datasets.X)
    negative_samples = filter(
        lambda sample: sample[2] is False, law_matching_datasets.X
    )

    assert len(list(positive_samples)) == len(list(negative_samples))


def test_resolve_reference(law_matching_datasets):
    acts = law_matching_datasets.acts
    reference = Reference(
        act="SARS-CoV-2-EindmaßnV",
        section_number="14",
        subsection_number="1",
        sentences="",
    )
    text = resolve_reference_to_subsection_text(
        reference, acts, datetime.date(2020, 4, 30)
    )
    assert (
        text
        == "Wissenschaftliche Bibliotheken und Archive dürfen unter Beachtung der Hygieneregeln nach § 2 "
        "Absatz 1 ab dem 27. April 2020 für den Leihbetrieb geöffnet werden."
    )


def test_saving_and_loading_csv_works(law_matching_datasets):

    with tempfile.NamedTemporaryFile() as file:

        law_matching_datasets.save_to_csv(file.name)

        new_datasets = LawMatchingDatasets.load_from_csv(file.name)

        assert (law_matching_datasets.X == new_datasets.X).all()
