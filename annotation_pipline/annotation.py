import re
from typing import Optional

DATE = re.compile(r"\d\d\.\d\d.\d\d")


class Annotation:
    """Takes an annotation from the fetched Hypothesis and exposes the relevant data."""

    def __init__(self, raw_annotation):
        self.raw_annotation = raw_annotation
        self.tags = raw_annotation["tags"]
        self.claim = raw_annotation["target"][0]["selector"][2]["exact"]
        self.id = raw_annotation["id"]
        self.reference = raw_annotation["text"]
        self.url = raw_annotation["uri"]

    @property
    def for_law_matching(self) -> bool:
        """An annotation if suitable for law matching if it is not tagged with on of the following tags."""
        no_law_matching_tags = {"no-context", "time-context", "no-law", "TODO"}
        return len(no_law_matching_tags & set(self.tags)) == 0

    @property
    def date(self) -> Optional[str]:
        for tag in self.tags:
            if date := DATE.search(tag):
                return date.group()
        return None

    @property
    def context(self) -> Optional[str]:
        for tag in self.tags:
            if "context:" in tag:
                return tag.split("context:")[1]
        return None

    @property
    def context_claim(self) -> str:
        if context := self.context:
            return f"{context} {self.claim}"
        return self.claim
