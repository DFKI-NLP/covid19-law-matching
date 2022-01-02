import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict

SECTION_DELIMITER = re.compile(r"\n\n\(\d\d?\)")
SECTION = re.compile(r"\n\n\((\d\w?)\) (.+?)(?=(\n\n\(\d\w?\)|$))", re.DOTALL)


def to_date(date_string: str) -> date:
    return datetime.strptime(date_string, "%d.%m.%Y").date()


@dataclass
class Subsection:
    subsection_number: str
    text: str
    section_number: str
    act: str


@dataclass
class Section:
    act: str
    section_number: str
    section_title: str
    text: str
    valid_from: date
    valid_to: date

    @classmethod
    def from_dict(cls, data, act):
        return cls(
            act=act,
            section_number=data["sectionNumber"],
            section_title=data["sectionTitle"],
            text=data["text"],
            valid_from=to_date(data["valid_from"]),
            valid_to=to_date(data["valid_to"]),
        )

    @property
    def subsections(self) -> Dict[str, Subsection]:
        if SECTION_DELIMITER.search(self.text):
            subsections = []
            for subsection_match in SECTION.finditer(self.text):
                subsection_number = subsection_match.group(1)
                text = subsection_match.group(2)
                subsections.append(
                    Subsection(subsection_number, text, self.section_number, self.act)
                )
        else:
            """Some laws are not split up in subsections, e.g. § 5 3. PflegeM-Cov-19-V
            (https://gesetze.berlin.de/bsbe/document/jlr-CoronaVPflege5VBEpP3)
            For those, we assign the subsection with the number 'full_section'"""
            subsections = [
                Subsection("full_section", self.text, self.section_number, self.act)
            ]
        return {subsection.subsection_number: subsection for subsection in subsections}
