import pytest
from bs4 import BeautifulSoup

from law_scraping.extract import LawSoup


@pytest.mark.skip
def test_law_soup():
    path = "/Users/Niklas/code/bachelor_thesis/law_scraping/data/html_pages/CoronaVVBE4rahmen_4020201129.html"
    file = open(path, "r")
    soup = LawSoup(BeautifulSoup(file.read(), "html.parser"))
    soup.extract_paragraphs()
    breakpoint()
