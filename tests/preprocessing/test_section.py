import datetime

from training.preprocessing.datasets_.models import Section


def test_subsection_numbers_contain_no_newlines(law_matching_datasets):

    section = law_matching_datasets.acts["SARS-CoV-2-EindV"].sections[1]

    for subsection in section.subsections.values():
        assert "\n" not in subsection.subsection_number


def test_subsection_numbers_can_go_above_9():
    section = Section.from_dict(
        {
            "sectionNumber": "7",
            "sectionTitle": "Verbote",
            "valid_from": "02.11.2020",
            "valid_to": "15.12.2020",
            "text": "\n\n(1) Tanzlustbarkeiten und ähnliche Unternehmen im Sinne der Gewerbeordnung in der Fassung der Bekanntmachung vom 22. Februar 1999 (BGBl. I S. 202), die zuletzt durch Artikel 5 des Gesetzes vom 19. Juni 2020 (BGBl. I S. 1403) geändert worden ist, dürfen nicht für den Publikumsverkehr geöffnet werden.\n\n(2) Gaststätten mit der besonderen Betriebsart Diskotheken und ähnliche Betriebe im Sinne des Gaststättengesetzes in der Fassung der Bekanntmachung vom 20. November 1998 (BGBl. I S. 3418), das zuletzt durch Artikel 14 des Gesetzes vom 10. März 2017 (BGBl. I S. 420) geändert worden ist, dürfen nicht für den Publikumsverkehr geöffnet werden. Sie dürfen Speisen und Getränke zur Abholung oder zur Lieferung anbieten. Für die Abholung sind geeignete Vorkehrungen zur Steuerung der Kaufabwicklung und zur Vermeidung von Menschenansammlungen zu treffen. Auch in anderen Gaststätten sind Tanzveranstaltungen nicht zulässig.\n\n(3) Fitness- und Tanzstudios, Saunen, Dampfbäder, Thermen und ähnliche Einrichtungen sind geschlossen zu halten. Satz 1 gilt auch für entsprechende Bereiche in Hotels und ähnlichen Einrichtungen.\n\n(4) Gaststätten im Sinne des Gaststättengesetzes in der Fassung der Bekanntmachung vom 20. November 1998 (BGBl. I S. 3418), das zuletzt durch Artikel 14 des Gesetzes vom 10. März 2017 (BGBl. I S. 420) geändert worden ist, dürfen nicht für den Publikumsverkehr geöffnet werden. Sie dürfen Speisen und Getränke zur Abholung oder zur Lieferung anbieten. Für die Abholung sind geeignete Vorkehrungen zur Steuerung der Kaufabwicklung und zur Vermeidung von Menschenansammlungen zu treffen. Satz 1 gilt nicht für den Betrieb von Kantinen.\n\n(5) Weihnachtsmärkte und Jahrmärkte sind verboten.\n\n(5a) Bei der Öffnung von Verkaufsstellen, Kaufhäusern und Einkaufszentren (Malls) gilt für die Steuerung des Zutritts und zur Sicherung des Mindestabstandes ein Richtwert von maximal einer Person (Kundinnen und Kunden) pro 10 Quadratmeter Verkaufsfläche und Geschäftsraum. Unterschreiten die Verkaufsfläche oder der Geschäftsraum eine Größe von 10 Quadratmeter, so darf jeweils maximal eine Kundin oder ein Kunde eingelassen werden. Aufenthaltsanreize dürfen nicht geschaffen werden. § 1 Absatz 4 gilt entsprechend.\n\n(6) Der Ausschank, die Abgabe und der Verkauf von alkoholischen Getränken sind in der Zeit von 23 Uhr bis 6 Uhr des Folgetages verboten.\n\n(7) Dienstleistungsgewerbe im Bereich der Körperpflege wie Kosmetikstudios, Massagepraxen, Tattoo-Studios und ähnliche Betriebe dürfen weder für den Publikumsverkehr geöffnet werden noch ihre Dienste anbieten. Satz 1 gilt nicht für Friseurbetriebe und medizinisch notwendige Behandlungen, insbesondere Physio-, Ergo- und Logotherapie, Podologie, Fußpflege und Heilpraktiker.\n\n(8) Kinos, Theater, Opern, Konzerthäuser, Museen, Gedenkstätten und kulturelle Veranstaltungsstätten in öffentlicher und privater Trägerschaft dürfen nicht für den Publikumsverkehr geöffnet werden. Der Leihbetrieb von Bibliotheken ist zulässig.\n\n(9) Vergnügungsstätten im Sinne der Baunutzungsverordnung in der Fassung der Bekanntmachung vom 21. November 2017 (BGBl. I S. 3786), Freizeitparks, Betriebe für Freizeitaktivitäten sowie Spielhallen, Spielbanken, Wettvermittlungsstellen und ähnliche Betriebe dürfen nicht für den Publikumsverkehr geöffnet werden.\n\n(10) Die Tierhäuser und das Aquarium des Zoologischen Gartens Berlin und die Tierhäuser des Tierparks Berlin-Friedrichsfelde dürfen nicht für den Publikumsverkehr geöffnet werden.\n\n(11) Touristische Übernachtungen in Hotels und anderen Beherbergungsbetrieben sind untersagt.\n\n(12) Prostitutionsgewerbe im Sinne des Prostituiertenschutzgesetzes vom 21. Oktober 2016 (BGBl. I S. 2372), das durch Artikel 57 des Gesetzes vom 20. November 2019 (BGBl. I S. 1626) geändert worden ist, dürfen weder für den Publikumsverkehr geöffnet werden, noch ihre Dienste außerhalb ihrer Betriebsstätte erbringen. Die Erbringung und Inanspruchnahme sexueller Dienstleistungen mit Körperkontakt und erotische Massagen sind untersagt.",
        },
        "SARS-CoV-2-Infektionsschutzverordnung",
    )
    assert len(section.subsections.keys()) == 12


def test_section_dates_are_correct(law_matching_datasets):

    act = law_matching_datasets.acts["SARS-CoV-2-Infektionsschutzverordnung"]
    sections = act.all_sections_for(datetime.date(2020, 12, 5))

    assert sections.valid_from == datetime.date(2020, 11, 29)
    assert sections.valid_to == datetime.date(2020, 12, 15)
