from datetime import date

from training.preprocessing.datasets_.models import Act


def test_act_from_file_constructor():
    file_path = "tests/fixtures/InfSchMV_fixture.json"
    act = Act.from_file(file_path)
    assert act


def test_act_should_return_valid_sections():
    file_path = "tests/fixtures/InfSchMV_fixture.json"
    act = Act.from_file(file_path)

    valid_date = date(2021, 2, 13)
    assert len(act.all_sections_for(valid_date)) == 32
    assert act.has_sections_for(valid_date)


def test_act_has_sections_for():
    file_path = "tests/fixtures/InfSchMV_fixture.json"
    act = Act.from_file(file_path)

    invalid_date = date(2022, 2, 13)

    assert not act.has_sections_for(invalid_date)
