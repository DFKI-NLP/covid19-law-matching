import pytest

from training.preprocessing.datasets_.models import Reference, parse_references


@pytest.mark.parametrize(
    "raw_string,target",
    [
        (
            "§ 10 (2) 3. InfSchMV",
            Reference(
                act="3. InfSchMV",
                section_number="10",
                subsection_number="2",
                sentences="",
            ),
        ),
        (
            "§ 2 (4) S. 2 Corona-ArbSchV",
            Reference(
                act="Corona-ArbSchV",
                section_number="2",
                subsection_number="4",
                sentences="S. 2",
            ),
        ),
        (
            "§ 4 (3) S. 1 Nr. 1 2. InfSchMV",
            Reference(
                act="2. InfSchMV",
                section_number="4",
                subsection_number="3",
                sentences="S. 1 Nr. 1",
            ),
        ),
        (
            "§ 6a (1) 2. InfSchMV",
            Reference(
                act="2. InfSchMV",
                section_number="6a",
                subsection_number="1",
                sentences="",
            ),
        ),
        (
            "§ 28 (2) InfSchMV",
            Reference(
                act="InfSchMV",
                section_number="28",
                subsection_number="2",
                sentences="",
            ),
        ),
    ],
)
def test_reference(raw_string, target):
    ref = Reference.from_string(raw_string)

    assert ref == target


def test_parse_references():
    raw_string = "§ 11 (3) 3. PflegeM-Cov-19-V i.V.m. § 12 (2) S. 4 3. PflegeM-Cov-19-V i.V.m. § 13 (1) S. 1 Nr. 1 3. PflegeM-Cov-19-V"
    refs = parse_references(raw_string)
    assert refs == [
        Reference(
            act="3. PflegeM-Cov-19-V",
            section_number="11",
            subsection_number="3",
            sentences="",
        ),
        Reference(
            act="3. PflegeM-Cov-19-V",
            section_number="12",
            subsection_number="2",
            sentences="S. 4",
        ),
        Reference(
            act="3. PflegeM-Cov-19-V",
            section_number="13",
            subsection_number="1",
            sentences="S. 1 Nr. 1",
        ),
    ]


def test_all_references():
    lines = open("tests/fixtures/annotation_references.txt", "r").readlines()
    raw_references = [line.strip() for line in lines]
    for raw_reference in raw_references:
        parse_references(raw_reference)
