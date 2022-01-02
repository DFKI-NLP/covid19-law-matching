import re
from dataclasses import dataclass
from typing import List

from .section import Subsection

ACT = re.compile(
    r"(2\. InfSchMV|3\. InfSchMV|InfSchMV|SchulHygCoV-19-VO"
    r"|Zweite Krankenhaus-Covid-19-Verordnung|Krankenhaus-Covid-19-Verordnung"
    r"|Zweite Pflegemaßnahmen-Covid-19-Verordnung|3\. PflegeM-Cov-19-V"
    r"|SARS-CoV-2-Infektionsschutzverordnung|IfSG|SARS-CoV-2-EindmaßnV|Corona-ArbSchV)"
)
SECTION_NR = re.compile(r"§ (\d\d?\S?) ")
SUBSECTION_NR = re.compile(r"\((\d\d?\S?)\)")


class ParsingError(Exception):
    pass


@dataclass
class Reference:
    act: str
    section_number: str
    subsection_number: str
    sentences: str

    @classmethod
    def from_string(cls, string: str):
        """Parse a reference string and returns Reference object"""
        act_match = ACT.search(string)
        if not act_match:
            raise ParsingError
        act = act_match.group(1)
        string = string.replace(
            act_match.group(0), ""
        )  # make the further parsing easier

        section_match = SECTION_NR.search(string)
        if not section_match:
            """Some laws are not split up into sections, e.g. § 17 SARS-CoV-2-EindmaßnV.
            For those we do not assign a section."""
            section_number = ""
        else:
            section_number = section_match.group(1)
            string = string.replace(
                section_match.group(0), ""
            )  # make the further parsing easier

        subsection_match = SUBSECTION_NR.search(string)
        if not subsection_match:
            """Some laws are not split up in subsections, e.g. § 5 3. PflegeM-Cov-19-V
            (https://gesetze.berlin.de/bsbe/document/jlr-CoronaVPflege5VBEpP3)
            For those, we do not assign a subsection."""
            subsection_number = ""
            sentences = string.strip()
        else:
            subsection_number = subsection_match.group(1)
            sentences = string.replace(
                subsection_match.group(0), ""
            ).strip()  # just the sentences remain
        return cls(act, section_number, subsection_number, sentences)

    def __eq__(self, other):
        if isinstance(other, Subsection):
            return (
                self.act == other.act
                and self.section_number == other.section_number
                and self.subsection_number == other.subsection_number
            )
        if isinstance(other, Reference):
            return (
                self.act == other.act
                and self.section_number == other.section_number
                and self.subsection_number == other.subsection_number
            )
        else:
            raise NotImplementedError


def parse_references(reference_annotation: str) -> List[Reference]:
    raw_refs = reference_annotation.split(" i.V.m. ")
    return [Reference.from_string(raw_string) for raw_string in raw_refs]
