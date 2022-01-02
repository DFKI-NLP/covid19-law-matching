import re

import pytest


def clean_string(string: str) -> str:
    string = (
        re.sub(r"\n\s*", " ", string)
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("  ", ".")
        .replace(" .", ".")
        .replace(" ,", ",")
        .strip()
    )
    string = re.sub(r"\?(?=\S)", "? ", string)
    string = re.sub(r":(?=\S)(?!innen)", ": ", string)
    string = re.sub(r"\s+", " ", string)
    string = re.sub(r":\s(?=innen)", ":", string)
    string = re.sub(r"(?<=\S),(?=\S)", ", ", string)
    return string


@pytest.mark.parametrize(
    "source,target",
    [
        ("Folgendes gilt:Etwas", "Folgendes gilt: Etwas"),
        ("Patient:innen", "Patient:innen"),
        ("Patient: innen", "Patient:innen"),
    ],
)
def test_colon_is_cleaned_correctly(source, target):
    assert target == clean_string(source)


@pytest.mark.parametrize(
    "source,target",
    [
        ("Zimmers,Besuch", "Zimmers, Besuch"),
        ("Zimmers ,Besuch", "Zimmers, Besuch"),
        ("Zimmers , Besuch", "Zimmers, Besuch"),
    ],
)
def test_commas_are_correctly_cleaned(source, target):
    assert target == clean_string(source)
