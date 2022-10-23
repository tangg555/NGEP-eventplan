"""
@Desc:
@Reference:
- GPT2LMHeadModel
https://huggingface.co/docs/transformers/model_doc/gpt2
@Notes:
- input_ids
token_type_ids (torch.LongTensor of shape (batch_size, input_ids_length), optional)
— Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:

"""

from typing import List, Dict
from pathlib import Path
import linecache

import torch
import numpy as np
from transformers import GPT2Tokenizer

from src.modules.datasets_base import BaseDataset
from src.modules.generation_models.datasets import (
    EventLineDataset,
    LeadingPlusEventDataset,
)


class HINTEventLineSbertOrderDataset(EventLineDataset):
    def __init__(self, tokenizer: GPT2Tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train",
                 hint_data_dir=None):
        if hint_data_dir is None:
            raise ("You should assign the datadir of thu-coai-hint to get files of sbert and order.")
        self.hint_data_dir = hint_data_dir
        self.sbert_score_file = Path(self.hint_data_dir).joinpath(f"{tgt_file_prefix}_sbertscore.target")
        self.order_file = Path(self.hint_data_dir).joinpath(f"{tgt_file_prefix}_order.target")
        if not self.sbert_score_file.exists() or not self.order_file.exists():
            raise FileNotFoundError(f"sbert_score_file: {self.sbert_score_file.exists()} ||"
                                    f"order_file: {self.order_file.exists()}")
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sen>"]})

    def _read_data(self):
        # changed to hint datadir
        self.tgt_file = Path(self.hint_data_dir).joinpath(f"{self.tgt_file_prefix}.target")
        # the customized requirement for hint
        self.src_data = replicate_data(self._read_clean_lines(self.src_file))
        self.tgt_data = self._read_clean_lines(self.tgt_file)
        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"data size of src_data {len(self.src_data)} should be equal to "
                             f"tgt_data {len(self.tgt_data)}")

    def __getitem__(self, index) -> Dict[str, str]:
        source_line = self.src_data[index]
        target_line = self.tgt_data[index]
        score_line = linecache.getline(str(self.sbert_score_file), index + 1).rstrip("\n")
        order_line = linecache.getline(str(self.order_file), index + 1).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        return {"src_text": source_line, "tgt_text": target_line, "data_id": index, "score": score_line,
                "order": order_line}

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

        normal_label, label, batch_order = [], [], []
        for x in batch:
            tmp = x["order"].split(",")
            label.append(int(tmp[0]))
            normal_label.append(int(int(tmp[0]) == 0))
            batch_order.append(list(map(int, tmp[1].split())))
        batch_encoding["type_labels"] = torch.tensor(label)
        batch_encoding["normal_labels"] = torch.tensor(normal_label)
        batch_encoding["orders"] = torch.tensor(batch_order)
        return batch_encoding


class HINTLeadingContextSbertOrderDataset(HINTEventLineSbertOrderDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train",
                 hint_data_dir=None):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix, hint_data_dir)


# for comparison experiments
class HINTLeadingPlusEventDataset(LeadingPlusEventDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 event_file_prefix="train_event",
                 tgt_file_prefix="train",
                 hint_data_dir=None):
        if hint_data_dir is None:
            raise ("You should assign the datadir of thu-coai-hint to get files of sbert and order.")
        self.hint_data_dir = hint_data_dir
        self.sbert_score_file = Path(self.hint_data_dir).joinpath(f"{src_file_prefix}_sbertscore.target")
        self.order_file = Path(self.hint_data_dir).joinpath(f"{src_file_prefix}_order.target")
        if not self.sbert_score_file.exists() or not self.order_file.exists():
            raise FileNotFoundError()
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, event_file_prefix, tgt_file_prefix)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sen>"]})

    def _read_data(self):
        # changed to hint datadir
        self.tgt_file = Path(self.hint_data_dir).joinpath(f"{self.tgt_file_prefix}.target")
        # the customized requirement for hint
        self.src_data = replicate_data(self._read_clean_lines(self.src_file))
        self.event_data = replicate_data(self._read_clean_lines(self.event_file))
        self.tgt_data = self._read_clean_lines(self.tgt_file)
        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"data size of src_data {len(self.src_data)} should be equal to "
                             f"tgt_data {len(self.tgt_data)}")

    def __getitem__(self, index) -> Dict[str, str]:
        source_line = self.src_data[index].rstrip("\n")
        event_line = self.event_data[index].rstrip("\n")
        target_line = self.tgt_data[index].rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert event_line, f"empty event line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        score_line = linecache.getline(str(self.sbert_score_file), index + 1).rstrip("\n")
        order_line = linecache.getline(str(self.order_file), index + 1).rstrip("\n")
        return {"src_text": source_line, "event_line": event_line, "tgt_text": target_line, "data_id": index,
                "score": score_line,
                "order": order_line}

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

        batch_score = []
        for x in batch:
            score_list = x["score"].split(",")
            sen_num = int(np.sqrt(len(score_list)))
            batch_score.append([float(s.split()[2]) for s in score_list])
        batch_encoding["sbert_score"] = torch.reshape(torch.tensor(batch_score), [len(batch_score), sen_num, sen_num])

        normal_label, label, batch_order = [], [], []
        for x in batch:
            tmp = x["order"].split(",")
            label.append(int(tmp[0]))
            normal_label.append(int(int(tmp[0]) == 0))
            batch_order.append(list(map(int, tmp[1].split())))
        batch_encoding["type_labels"] = torch.tensor(label)
        batch_encoding["normal_labels"] = torch.tensor(normal_label)
        batch_encoding["orders"] = torch.tensor(batch_order)
        return batch_encoding


def replicate_data(data: list):
    new_ = []
    for one in data:
        new_.append(one)
        new_.append(one)
    return new_
