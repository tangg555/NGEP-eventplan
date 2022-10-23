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

from torch.utils.data import Dataset
from src.modules.datasets_base import BaseDataset
from preprocessing.generation_models.event_ontology import EventGraph
from src.utils.string_utils import rm_extra_spaces

class EventPlanDataset(Dataset):
    leading_suffix = "source.txt"
    event_suffix = "source.txt"
    src_suffix = leading_suffix
    tgt_suffix = event_suffix

    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_event_length,
                 leading_file_prefix="train",
                 event_file_prefix="train_event",
                 device=None
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.leading_file_prefix = leading_file_prefix
        self.event_file_prefix = event_file_prefix
        self.leading_file = Path(data_dir).joinpath(f"{self.leading_file_prefix}.{self.leading_suffix}")
        self.event_file = Path(data_dir).joinpath(f"{self.event_file_prefix}.{self.event_suffix}")
        self.src_file = self.leading_file
        self.tgt_file = self.event_file
        self.src_data: List[str] = None
        self.tgt_data: List[str] = None
        self.max_source_length = max_source_length
        self.max_target_length = max_event_length

        self.pad_token_id = self.tokenizer.pad_token_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._read_data()
        self.tokenizer.add_special_tokens({'additional_special_tokens':
                                               [EventGraph.event_s, EventGraph.event_sep, EventGraph.event_e],
                                           })

    def _read_clean_lines(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = rm_extra_spaces(line)
                if len(line) > 0:
                    data.append(line)
        return data

    def __len__(self):
        return len(self.src_data)

    def _read_data(self):
        """
        - make data
        leading context: [MALE] 's parents were overweight .
        event line: [EVENT_s] was crisp [EVENT_sep] felt good [EVENT_sep] decided keep [EVENT_sep] went [EVENT_e]
        result:
        [MALE] 's parents were overweight . [EVENT_s] -> was crisp
        [MALE] 's parents were overweight . [EVENT_s] was crisp [EVENT_sep] -> felt good
        ......
        [MALE] 's parents were overweight . [EVENT_s] was crisp [EVENT_sep] felt good [EVENT_sep] decided keep
        [EVENT_sep] -> went
        """
        self.src_data = []
        self.tgt_data = []
        leading_data = self._read_clean_lines(self.leading_file)
        event_data = self._read_clean_lines(self.event_file)
        for leading_context, event_text in zip(leading_data, event_data):
            events = event_text.strip().replace(EventGraph.event_s, "").replace(EventGraph.event_e, "").split(EventGraph.event_sep)
            for idx in range(1, len(events)):
                serialised_events = EventGraph.event_s + " " + f"{EventGraph.event_sep}".join(events[:idx])  \
                                    + " " + EventGraph.event_e
                src_text = f"{leading_context.strip()} {serialised_events}"
                tgt_text = f"{events[idx]}" # the next event
                self.src_data.append(src_text)
                self.tgt_data.append(tgt_text)
        assert len(self.src_data) == len(self.tgt_data)

    def __getitem__(self, index):
        src_line = self.src_data[index].rstrip("\n")
        tgt_line = self.tgt_data[index].rstrip("\n")
        assert src_line, f"empty source line for index {index}"
        assert tgt_line, f"empty event line for index {index}"
        return {"src_text": src_line, "tgt_text": tgt_line, "data_id": index}


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

