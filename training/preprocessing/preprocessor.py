from typing import List, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .datasets_ import LawMatchingSample

Offset = Tuple[int, int]
RawClaimExtractionDataset = NDArray[Tuple[str, List[Offset]]]
RawLawMatchingDataset = NDArray[LawMatchingSample]


class Input(TypedDict, total=False):
    input_ids: List[int]
    offset_mapping: List[Offset]
    attention_mask: List[int]
    labels: NDArray[int]


class CustomDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.X["input_ids"][idx],
            "attention_mask": self.X["attention_mask"][idx],
            "labels": self.X["labels"][idx],
        }


class Preprocessor:
    """
    Takes raw data from a *Datasets class, tokenizes is, and returns a torch Dataset.
    Usage:
        datasets = ClaimExtractionDatasets.from_database()
        preprocessor = Preprocessor(task='claim_extraction', tokenizer=AutoTokenizer.from_pretrained(checkpoint))
        inputs = preprocessor(datasets.X)
        output = model(**inputs[0])
    """

    def __init__(self, tokenizer, task: str):
        self.tokenizer = tokenizer
        self.task = task

    def preprocess_claim_extraction(
        self, X: RawClaimExtractionDataset
    ) -> CustomDataset:
        """
        Transforms a the raw data (text string with list of claim (start, end) tuples),
        tokenizes them, aligns the claim indexes with the tokenized text, and returns
        a torch Dataset.
        """
        samples_texts = []
        sample_offsets = []
        for sample_text, claim_offsets in X:
            samples_texts.append(sample_text)
            sample_offsets.append(claim_offsets)

        tokenized_inputs = self.tokenizer(
            samples_texts,
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        tokenized_inputs["labels"] = []
        for i in range(len(tokenized_inputs["input_ids"])):
            input_ids = tokenized_inputs["input_ids"][i]
            offset_mapping = tokenized_inputs["offset_mapping"][i]
            claim_offsets = sample_offsets[i]
            tokenized_inputs["labels"].append(
                self.align_claim_labels(
                    input_ids,
                    offset_mapping,
                    claim_offsets,
                    self.tokenizer.pad_token_id,
                ),
            )
        return CustomDataset(tokenized_inputs)

    @staticmethod
    def align_claim_labels(
        input_ids: List[int],
        offset_mapping: List[Offset],
        claim_offsets: List[Offset],
        pad_token_id: int,
    ) -> List[int]:
        """Takes a list of input ids, offset mapping from the tokenizer, and (claim_start, claim_end) tuples,
        that mark the offset for the original text. Returns a list with labels in BIO schema for the tokenized text."""
        labels = np.zeros(len(input_ids))

        def get_offset(original_position: int) -> int:
            # Does not work if the original claim started or ended with a space, since those don't have an offset
            for index, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_end == 0:
                    # special token
                    continue
                if offset_start <= original_position < offset_end:
                    return index

        for claim_start, claim_end in claim_offsets:
            start = get_offset(claim_start)
            end = get_offset(claim_end - 1)
            if end is None:
                raise Exception(
                    "The respective chunk is too big, so the input got truncated."
                )
            labels[start] = 1
            labels[slice(start + 1, end + 1)] = 2
        labels[
            np.array(input_ids) == pad_token_id
        ] = -100  # to exclude pad tokens from loss prediction

        return labels.astype(int).tolist()

    def preprocess_law_matching(self, X: RawLawMatchingDataset) -> CustomDataset:

        samples = {"input_ids": [], "attention_mask": [], "labels": []}
        for claim, text, is_match in X:
            sample = dict(**self.tokenizer(claim, text, truncation=True))
            samples["input_ids"].append(sample["input_ids"])
            samples["attention_mask"].append(sample["attention_mask"])
            samples["labels"].append(int(is_match))
        return CustomDataset(samples)

    def __call__(self, data: NDArray) -> Dataset:
        if self.task == "claim_extraction":
            return self.preprocess_claim_extraction(data)
        if self.task == "law_matching":
            return self.preprocess_law_matching(data)
        else:
            raise NotImplementedError
