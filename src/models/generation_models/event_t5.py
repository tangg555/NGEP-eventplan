"""
@Desc:
@Reference:
- t5
https://huggingface.co/docs/transformers/model_doc/t5
@Notes:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config
from transformers import T5Tokenizer

from src.utils.file_utils import save_json, pickle_save
from src.modules.generation_models.datasets import (
    LeadingContextDataset,
    EventLineDataset,
    LeadingToEventsDataset,
    LeadingPlusEventDataset,
)
from src.utils.generation_models import model_utils
from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.models.generation_models.event_bart import EventBart, LeadingContextBart, LeadingPlusEventBart

logger = logging.getLogger(__name__)

class LeadingContextT5(LeadingContextBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # config
        self.config: T5Config = T5Config.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model: T5ForConditionalGeneration = \
            self._load_model(self.hparams.model_name_or_path, T5ForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
        self.dataset_class = LeadingContextDataset

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _step(self, batch: dict):
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        decoder_input_ids = self._shift_right_t5(tgt_ids)

        if self.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch_fn(batch)

        outputs = self(src_ids,
                       attention_mask=src_mask,
                       decoder_input_ids=decoder_input_ids,
                       use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            # lm_ligits: [batch, seq, vocab] tgt_ids: [batch, seq]
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            losses_ = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=self.pad_token_id
            )
        lm_loss = loss
        return lm_loss

    @torch.no_grad()
    def sample_sequence(self, batch, use_top_p=False, top_p=0.9):
        batch_size = len(batch["ids"])
        # Note that T5 uses the pad_token_id as the decoder_start_token_id
        decoder_input_ids = torch.tensor([self.tokenizer.pad_token_id for _
                                          in range(batch_size)])[:, None].to(self.device)
        for _ in range(self.hparams.max_target_length):
            outputs = self(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           decoder_input_ids=decoder_input_ids,
                           use_cache=False, return_dict=True)
            logits = outputs["logits"]
            logits = logits[:, -1, :]
            if use_top_p:
                logits = top_p_logits(logits, p=top_p, device=self.device)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.topk(input=probs, k=1).indices
            decoder_input_ids = torch.cat([decoder_input_ids, pred], 1)
            # early stop
            if pred[:, 0].eq(self.tokenizer.eos_token_id).sum() == pred.shape[0]:
                break
        generated_ids = decoder_input_ids
        return generated_ids

class EventT5(LeadingContextT5):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = EventLineDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> EventLineDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()

        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=f"{src_file_prefix}"
                            f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

class LeadingToEventsT5(LeadingContextT5):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingToEventsDataset

    def ids_to_clean_string(self, token_list, tokenizer):
        # leading to events is special, because we need special tokens of EventGraph
        real_s = 0
        for index_, token_ in enumerate(token_list):
            if token_ not in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
                real_s = index_
                break
        token_list = token_list[real_s:]
        string = tokenizer.decode(token_list, skip_special_tokens=False)
        string = string[:string.find("</s>")].strip()
        for one in [tokenizer.bos_token, tokenizer.eos_token]:
            string = string.replace(one, " ")
        string = " ".join([one for one in string.split(" ")])
        return string

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train_event", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val_event", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test_event", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)


class LeadingPlusEventT5(LeadingContextT5):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingPlusEventDataset
        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> LeadingPlusEventDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            event_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)
