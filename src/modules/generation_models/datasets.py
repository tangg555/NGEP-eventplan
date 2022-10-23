"""
@Desc:
@Reference:
- transformers examples for using BART model
https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization
https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/2
- add_special_tokens
https://huggingface.co/docs/transformers/v4.17.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
- linecache
https://blog.csdn.net/my2010Sam/article/details/38022041
- torch Dataset
https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
@Notes:
- add_special_tokens
special_tokens_dict (dictionary str to str or tokenizers.AddedToken) —
Keys should be in the list of predefined special attributes:
[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
Tokens are only added if they are not already in the vocabulary (tested by checking
if the tokenizer assign the index of the unk_token to them).
- collate_fn
A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch.
See this section on more about collate_fn.
"""

from typing import List
from pathlib import Path
import linecache

import numpy as np
import torch

from src.modules.datasets_base import BaseDataset
from preprocessing.event_plan.event_ontology import EventGraph
from src.utils.string_utils import rm_extra_spaces


class EventLineDataset(BaseDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                                               [EventGraph.event_s, EventGraph.event_sep, EventGraph.event_e],
                                           })

    def __getitem__(self, index):
        source_line = self.src_data[index]
        target_line = self.tgt_data[index]
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        return {"src_text": source_line, "tgt_text": target_line, "data_id": index}

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding

    def __len__(self):
        return len(self.src_data)


class LeadingContextDataset(EventLineDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)


class LeadingToEventsDataset(LeadingContextDataset):
    tgt_suffix = "source.txt"

    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)


class LeadingEventDataset(EventLineDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 event_file_prefix="train_event",
                 tgt_file_prefix="train", ):
        self.event_file = Path(data_dir).joinpath(f"{event_file_prefix}.{self.src_suffix}")
        self.event_data: List[str] = None
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)

    def _read_data(self):
        self.src_data = self._read_clean_lines(self.src_file)
        self.event_data = self._read_clean_lines(self.event_file)
        self.tgt_data = self._read_clean_lines(self.tgt_file)
        assert len(self.src_data) == len(self.tgt_data)

    def __getitem__(self, index):
        source_line = self.src_data[index].rstrip("\n")
        event_line = self.event_data[index].rstrip("\n")
        target_line = self.tgt_data[index].rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert event_line, f"empty event line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        return {"src_text": source_line, "event_line": event_line, "tgt_text": target_line, "data_id": index}

    def collate_fn(self, batch):
        batch_encoding = {}
        batch_encoding["leading_contexts"] = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        batch_encoding["event_lines"] = self.tokenizer(
            [x["event_line"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding


# for comparison experiments
class LeadingPlusEventDataset(LeadingEventDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 event_file_prefix="train_event",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, event_file_prefix, tgt_file_prefix)

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [f'{x["src_text"]} {x["event_line"]}' for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding


# =================================== sbert datasets ==========================================
class LeadingSbertDataset(LeadingContextDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train",
                 sbert_data_dir=None):
        if sbert_data_dir is None:
            raise ("You should assign the datadir of sbert_data_dir")
        self.sbert_data_dir = sbert_data_dir
        self.sbert_score_file = Path(self.sbert_data_dir).joinpath(f"{tgt_file_prefix}_sbertscore.target")
        if not self.sbert_score_file.exists():
            raise FileNotFoundError(f"sbert_score_file: {self.sbert_score_file.exists()}")
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sen>"]})

    def _clip_data(self, data: list):
        new_ = []
        for idx in range(0, len(data), 2):
            new_.append(data[idx])
        return new_

    def _read_data(self):
        # changed to hint datadir
        self.tgt_file = Path(self.sbert_data_dir).joinpath(f"{self.tgt_file_prefix}.target")
        # the customized requirement for hint
        self.src_data = self._read_clean_lines(self.src_file)
        self.tgt_data = self._clip_data(self._read_clean_lines(self.tgt_file))
        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"data size of src_data {len(self.src_data)} should be equal to "
                             f"tgt_data {len(self.tgt_data)}")

    def __getitem__(self, index):
        source_line = self.src_data[index].rstrip("\n")
        target_line = self.tgt_data[index].rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        # sbert data size is two times of other data
        score_line = linecache.getline(str(self.sbert_score_file), index * 2 + 1).rstrip("\n")
        return {"src_text": source_line, "tgt_text": target_line, "data_id": index,
                "score": score_line}

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]

        batch_score = []
        for x in batch:
            score_list = x["score"].split(",")
            sen_num = int(np.sqrt(len(score_list)))
            batch_score.append([float(s.split()[2]) for s in score_list])
        batch_encoding["sbert_score"] = torch.reshape(torch.tensor(batch_score), [len(batch_score), sen_num, sen_num])
        return batch_encoding


class EventSbertDataset(LeadingSbertDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train",
                 sbert_data_dir=None):
        super().__init__(tokenizer, data_dir, max_source_length,
                         max_target_length, src_file_prefix, tgt_file_prefix, sbert_data_dir)


class LeadingEventSbertDataset(LeadingEventDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 event_file_prefix="train_event",
                 tgt_file_prefix="train",
                 sbert_data_dir=None):
        if sbert_data_dir is None:
            raise ("You should assign the datadir of sbert_data_dir")
        self.sbert_data_dir = sbert_data_dir
        self.sbert_score_file = Path(self.sbert_data_dir).joinpath(f"{tgt_file_prefix}_sbertscore.target")
        if not self.sbert_score_file.exists():
            raise FileNotFoundError(f"sbert_score_file: {self.sbert_score_file.exists()}")
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, event_file_prefix, tgt_file_prefix)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sen>"]})

    def _clip_data(self, data: list):
        new_ = []
        for idx in range(0, len(data), 2):
            new_.append(data[idx])
        return new_

    def _read_data(self):
        # changed to hint datadir
        self.tgt_file = Path(self.sbert_data_dir).joinpath(f"{self.tgt_file_prefix}.target")
        # the customized requirement for hint
        self.src_data = self._read_clean_lines(self.src_file)
        self.event_data = self._read_clean_lines(self.event_file)
        self.tgt_data = self._clip_data(self._read_clean_lines(self.tgt_file))
        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"data size of src_data {len(self.src_data)} should be equal to "
                             f"tgt_data {len(self.tgt_data)}")

    def __getitem__(self, index):
        source_line = self.src_data[index].rstrip("\n")
        event_line = self.event_data[index].rstrip("\n")
        target_line = self.tgt_data[index].rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert event_line, f"empty event line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        # sbert data size is two times of other data
        score_line = linecache.getline(str(self.sbert_score_file), index * 2 + 1).rstrip("\n")
        return {"src_text": source_line, "event_line": event_line, "tgt_text": target_line, "data_id": index,
                "score": score_line,
                }

    def collate_fn(self, batch):
        batch_encoding = {}
        batch_encoding["leading_contexts"] = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        batch_encoding["event_lines"] = self.tokenizer(
            [x["event_line"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]

        batch_score = []
        for x in batch:
            score_list = x["score"].split(",")
            sen_num = int(np.sqrt(len(score_list)))
            batch_score.append([float(s.split()[2]) for s in score_list])
        batch_encoding["sbert_score"] = torch.reshape(torch.tensor(batch_score), [len(batch_score), sen_num, sen_num])
        return batch_encoding
