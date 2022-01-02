import json
from dataclasses import dataclass
from typing import List, Dict
import datetime

from .section import Section


@dataclass
class Act:
    abbreviation: str
    full_name: str
    sections: List[Section]

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        sections = [
            Section.from_dict(data=section_dict, act=data["abbreviation"])
            for section_dict in data["sections"]
        ]
        return cls(
            abbreviation=data["abbreviation"], full_name=data["name"], sections=sections
        )

    def all_sections_for(self, date: datetime.date) -> Dict[str, Section]:
        return {
            section.section_number: section
            for section in filter(
                lambda section: section.valid_from <= date <= section.valid_to,
                self.sections,
            )
        }

    def has_sections_for(self, date: datetime.date) -> bool:
        """Returns True if there are valid sections for a certain date."""

        return len(self.all_sections_for(date)) > 0

    def __repr__(self):
        return f"Act({self.full_name} ({self.abbreviation}))"
