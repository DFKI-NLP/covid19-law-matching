import sqlite3
import csv
from operator import itemgetter
from itertools import groupby
from typing import List, Tuple

import spacy
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, ShuffleSplit

ClaimExtractionSample = Tuple[str, List[Tuple[int, int]]]
Offset = Tuple[int, int]


class ClaimExtractionDatasets:
    """Loads the claim extraction data from the db, groups it, and returns folds
    Usage:
        datasets = ClaimExtractionDatasets.load_from_database()
        for train, test in datasets.folds:
            # preprocess train and test set

    or (without folds):
        datasets = ClaimExtractionDatasets.load_from_database()
        train = datasets.train
        test = datasets.test
    """

    TASK = "claim_extraction"

    def __init__(self, rows, folds, seed=0):
        self.kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        self.X: List[ClaimExtractionSample] = []
        grouped_rows = self.group_rows(rows)
        nlp = spacy.load("de_core_news_sm")
        for sample_text, claim_offsets in grouped_rows:
            chunks = self.chunk_fulltext(sample_text, claim_offsets, nlp)
            for sample in chunks:
                self.X.append(sample)

        self.X = self.limit_samples_without_claims(self.X)
        self.X = np.array(self.X, dtype=object)
        self.train_split, self.test_split = next(
            ShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(self.X)
        )

    @property
    def folds(self):
        for train_split, test_split in self.kf.split(self.X):
            yield self.X[train_split], self.X[test_split]

    @property
    def train(self) -> NDArray[ClaimExtractionSample]:
        return self.X[self.train_split]

    @property
    def test(self) -> NDArray[ClaimExtractionSample]:
        return self.X[self.test_split]

    @classmethod
    def group_rows(cls, rows) -> NDArray[ClaimExtractionSample]:
        """Takes the db rows (url, fulltext, claim), groups them by url, and returns for every
        url a tuple (fulltext, [(start, end)]) with the claim offsets."""
        sorted_rows = sorted(rows, key=itemgetter(0))
        groups = groupby(sorted_rows, key=itemgetter(0))
        return_list = []
        for url, group in groups:
            group = list(group)
            fulltext: str = group[0][1]
            article = (fulltext, [])
            for _, _, claim in group:
                claim = claim.strip()
                if claim in fulltext:
                    start: int = fulltext.find(claim)
                    end: int = start + len(claim)
                    article[1].append((start, end))
            if article[0] and article[1]:
                return_list.append(article)

        return np.array(return_list, dtype=object)

    @staticmethod
    def chunk_fulltext(
        fulltext: str, claims: List[Offset], nlp: spacy.Language
    ) -> List[Tuple[str, List[Offset]]]:
        """Split article fulltext into smaller chunks (max. 512 tokens), with the condition that
        every claim is fully contained in one chunk."""
        max_length = 2200  # this is a heuristic
        sents = nlp(fulltext).sents
        chunks = []
        return_list = []

        current_chunk = ""
        for sentence in sents:
            if len(current_chunk) > max_length:
                chunk_start: int = fulltext.find(current_chunk)
                chunk_end: int = chunk_start + len(current_chunk)
                for claim_start, claim_end in claims:
                    if claim_start < chunk_end < claim_end:
                        # we found a overlapping claim. continue making a chunk
                        break
                else:
                    # no overlapping claims! let's move on to the next chunk
                    chunks.append(current_chunk)
                    current_chunk = ""
            current_chunk += " " + str(sentence)
        chunks.append(current_chunk)

        for chunk in chunks:
            chunk_claim_offsets = []
            for claim_start, claim_end in claims:
                claim = fulltext[slice(claim_start, claim_end)]
                if claim in chunk:
                    start: int = chunk.find(claim)
                    end: int = start + len(claim)
                    chunk_claim_offsets.append((start, end))
            return_list.append((chunk, chunk_claim_offsets))

        return return_list

    @classmethod
    def limit_samples_without_claims(cls, X):
        X_with_claims = filter(lambda sample: len(sample[1]) > 0, X)
        return_claim = list(X_with_claims)
        X_without_claims = filter(lambda sample: len(sample[1]) == 0, X)
        num_samples_without_claims = int(len(return_claim) * 0.00)
        for _ in range(num_samples_without_claims):
            return_claim.append(next(X_without_claims))
        return return_claim

    @classmethod
    def load_from_database(
        cls,
        database="data/database.db",
        folds=5,
        seed=0,
    ):

        connection = sqlite3.connect(database)

        cursor = connection.cursor()
        cursor.execute(
            "SELECT f.url, plaintext, claim FROM fulltext f INNER JOIN claims c on f.url=c.url"
        )
        rows = cursor.fetchall()
        cursor.close()
        return cls(rows, folds=folds, seed=seed)

    def save_to_csv(self, file_name):
        with open(file_name, "w+", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            for sample in self.X:
                writer.writerow(sample)

    def save_to_disk(self, file_path, with_healtcheck=False):
        with open(file_path, "w+") as file:
            writer = csv.writer(file)
            for row in self.X:
                writer.writerow(row)
                if with_healtcheck:
                    text, offsets = row
                    ltext = list(text)
                    for offset in offsets:
                        for i in range(*offset):
                            ltext[i] = "_"
                    writer.writerow(["".join(ltext), ""])
