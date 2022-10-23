"""
@Desc:
@Reference:
- GPT2LMHeadModel
https://huggingface.co/docs/transformers/model_doc/gpt2
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

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer

from src.modules.generation_models.gpt2_datasets import (
    GPT2LeadingContextDataset,
    GPT2EventLineDataset,
    GPT2LeadingToEventsDataset,
    GPT2LeadingPlusEventDataset,
)
from src.utils.generation_models import model_utils
from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.models.generation_models.event_bart import EventBart, LeadingContextBart, LeadingPlusEventBart

logger = logging.getLogger(__name__)


class LeadingContextGPT2(LeadingContextBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # config
        self.config: GPT2Config = GPT2Config.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, GPT2LMHeadModel, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
        self.dataset_class = GPT2LeadingContextDataset

    def _step(self, batch: dict):
        if self.save_readable_batch and not self.already_saved_batch:
            self.save_readable_batch_fn(batch)
        outputs = self(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       token_type_ids=batch["token_type_ids"],
                       labels=batch["input_ids"],
                       return_dict=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            losses_ = outputs["loss"]
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, batch["input_ids"][..., 1:].contiguous(),
                self.hparams.label_smoothing, ignore_index=self.pad_token_id
            )
        lm_loss = loss
        return lm_loss

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["tokens_per_batch"] = batch["input_ids"].ne(self.pad_token_id).sum()
        logs["batch_size"] = batch["input_ids"].shape[0]
        logs["source_pad_tokens_num"] = batch["input_ids"].eq(self.pad_token_id).sum()
        logs["source_pad_tokens_ratio"] = batch["input_ids"].eq(self.pad_token_id).float().mean()
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def sample_sequence(self, batch, use_top_p=False, top_p=0.9):
        # eval_batch_size must be 1, otherwise the output will be wrong.
        # pad = endoftext, the generated text will be endoftext.
        batch_size = len(batch["ids"])
        generated_ids = None
        input_ids = batch["src_ids"]
        attention_mask = batch["src_attention_mask"]
        eos_counter = torch.zeros([batch_size]).to(self.device)
        for _ in range(self.hparams.max_target_length):
            # input size limitation
            if input_ids.shape[1] >= self.hparams.max_source_length:
                end_ids = torch.ones(input_ids.shape[0]).type_as(generated_ids). \
                              to(self.device) * self.tokenizer.eos_token_id
                generated_ids = torch.cat([generated_ids, end_ids[:, None]], dim=1)
                break
            outputs = self(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=True)
            logits = outputs["logits"]
            # 这里采集的位置有问题
            logits = logits[:, -1, :]
            if use_top_p:
                logits = top_p_logits(logits, p=top_p, device=self.device)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.topk(input=probs, k=1).indices

            if generated_ids is None:
                generated_ids = pred
            else:
                generated_ids = torch.cat([generated_ids, pred], dim=1)
            input_ids = torch.cat([input_ids, pred], 1)
            attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).to(self.device)], dim=1)
            # early stop
            eos_counter += pred[:, 0].eq(self.tokenizer.eos_token_id)
            if eos_counter.ge(1).sum() == batch_size:
                break
        return generated_ids

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        batch["labels"] = batch["tgt_ids"]
        if fast_generate:
            print(f"fast_generate is not supported for {self.model_name}")
        return super()._generative_step(batch, fast_generate=False)


class EventGPT2(LeadingContextGPT2):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = GPT2EventLineDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> GPT2EventLineDataset:
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

class LeadingToEventsGPT2(LeadingContextGPT2):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = GPT2LeadingToEventsDataset

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


class LeadingPlusEventGPT2(LeadingContextGPT2):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = GPT2LeadingPlusEventDataset
        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> GPT2LeadingPlusEventDataset:
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
