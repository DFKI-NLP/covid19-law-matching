import spacy

from training.preprocessing.datasets_ import LawMatchingDatasets


def test_baseline():
    nlp = spacy.load("de_core_news_sm")
    datasets = LawMatchingDatasets.load_from_database()
    doc = "An öffentlichen Schulen und Schulen in freier Trägerschaft einschließlich der Einrichtungen des Zweiten Bildungsweges und der Angebote der außerunterrichtlichen und ergänzenden Förderung und Betreuung darf vorbehaltlich der Absätze 6 und 7 kein Lehr- und Betreuungsbetrieb in Präsenz stattfinden. Abweichungen von Satz 1 zum Zwecke einer an das Infektionsgeschehen angepassten Wiederaufnahme des Lehr- und Betreuungsbetriebs in Präsenz bestimmt die für Bildung zuständige Senatsverwaltung durch Rechtsverordnung nach § 25 Absatz 1 und 2."
    breakpoint()
