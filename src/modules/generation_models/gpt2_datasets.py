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

from typing import List
from pathlib import Path
import torch
from transformers import GPT2Tokenizer

from src.modules.datasets_base import BaseDataset
from src.modules.generation_models.datasets import (
    EventLineDataset,
    LeadingPlusEventDataset,
)


class GPT2EventLineDataset(EventLineDataset):
    def __init__(self, tokenizer: GPT2Tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        src_batch = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        tgt_batch = self.tokenizer(
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        batch_encoding["src_ids"] = src_batch["input_ids"]
        batch_encoding["src_attention_mask"] = src_batch["attention_mask"]
        batch_encoding["tgt_ids"] = tgt_batch["input_ids"]
        token_type_ids = batch_encoding["attention_mask"].clone()
        token_type_ids[:, :batch_encoding["src_attention_mask"].shape[1]] -= batch_encoding["src_attention_mask"]
        batch_encoding["token_type_ids"] = token_type_ids
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding


class GPT2LeadingContextDataset(GPT2EventLineDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)


class GPT2LeadingToEventsDataset(GPT2LeadingContextDataset):
    tgt_suffix = "source.txt"

    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)


# for comparison experiments
class GPT2LeadingPlusEventDataset(LeadingPlusEventDataset):
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
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        src_batch = self.tokenizer(
            [f'{x["src_text"]} {x["event_line"]}' for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        tgt_batch = self.tokenizer(
            [x["tgt_text"] + f" {self.tokenizer.eos_token}" for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        batch_encoding["src_ids"] = src_batch["input_ids"]
        batch_encoding["src_attention_mask"] = src_batch["attention_mask"]
        batch_encoding["tgt_ids"] = tgt_batch["input_ids"]
        token_type_ids = batch_encoding["attention_mask"].clone()
        token_type_ids[:, :batch_encoding["src_attention_mask"].shape[1]] -= batch_encoding["src_attention_mask"]
        batch_encoding["token_type_ids"] = token_type_ids
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding
