"""
@Desc:
@Reference:
- thu-coai-hint
https://arxiv.org/pdf/2105.08963v1.pdf
@Notes:
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.utils.file_utils import save_json, pickle_save
from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.modules.generation_models.hint_datasets import (
    HINTEventLineSbertOrderDataset,
    HINTLeadingContextSbertOrderDataset,
    HINTLeadingPlusEventDataset
)
from src.models.generation_models import (
    LeadingContextBart,
    LeadingPlusEventBart,
)
from src.utils.generation_models import model_utils
from src.configuration.constants import BASE_DIR
from src.utils import nlg_eval_utils

logger = logging.getLogger(__name__)


class LeadingContextHINT(LeadingContextBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = HINTLeadingContextSbertOrderDataset

        self.reorder_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)
        self.sbert_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)


    def save_readable_batch_fn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = dict()
        for key, val in batch.items():
            if key in ["input_ids", "input_ids", "decoder_input_ids"]:
                readable_batch.update({key: self.tokenizer.batch_decode(val.tolist())})
            elif key in ["ids"]:
                readable_batch.update({key: val})
            elif key in ["orders"]:
                readable_batch.update({key: val.tolist()})
        save_json(readable_batch, Path(self.experiment_output_dir) / "text_batch.json")
        self.already_saved_batch = True
        return readable_batch

    def _step(self, batch: dict) -> Tuple:
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        decoder_input_ids = shift_tokens_right(tgt_ids,
                                               self.pad_token_id,
                                               self.decoder_start_token_id)
        if self.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch_fn(batch)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            loss_mask = (batch["normal_labels"][:, None] * torch.ones_like(tgt_ids) * (
                    1 - tgt_ids.eq(self.tokenizer.pad_token_id).to(torch.float))).view(-1)
            # lm_ligits: [batch, seq, vocab] tgt_ids: [batch, seq]
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            loss = torch.sum(self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1)) * loss_mask) / (
                    torch.sum(loss_mask) + 1e-20)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=self.pad_token_id
            )
        lm_loss = loss + 0.
        reorder_loss, sbert_loss = torch.tensor(0., ), torch.tensor(0., )

        # [batch_size, sequence_length, hidden_size]
        hidden_states = outputs["decoder_hidden_states"][-1]
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # [batch_size, sequence_length]
        sen_pos = batch["labels"].eq(self.tokenizer.mask_token_id).to(torch.float)
        dis_pos = torch.cat([torch.zeros([batch_size, 1]).to(sen_pos.device), sen_pos[:, :-1]], 1)
        # coeff = 1. / (float(hparams.reorder) + float(hparams.sbert))

        # # if hparams.reorder:
        # sen_idx = dis_pos.nonzero()
        #
        # # [bath_size, sen_length]
        # reorder_label = batch["orders"]
        # sentence_num = reorder_label.size()[1]
        #
        # # [batch_size, sentence_num, hidden_size]
        # sent_hidden_states_gather = self.gather_nd(hidden_states, sen_idx)
        # sent_hidden_states = torch.reshape(sent_hidden_states_gather, [batch_size, sentence_num, hidden_size])
        #
        # self.reorder_linear_layer = self.reorder_linear_layer.float()
        #
        # # [batch_size, sentence_num, sentence_num]
        # sen_att_logits = torch.matmul(self.reorder_linear_layer(sent_hidden_states),
        #                               torch.transpose(sent_hidden_states, 1, 2))
        #
        # # [batch_size, sentence_num, sentence_num]
        # pred_score = torch.sigmoid(sen_att_logits)  # + torch.transpose(sen_att_logits, 1, 2))
        # reorder_mask = (1 - torch.eye(sentence_num)[None, :, :].to(batch["type_labels"].device)) * \
        #                (torch.eq(batch["type_labels"], 0) | torch.eq(batch["type_labels"], 1)).to(torch.float)[:,
        #                None, None]
        #
        # true_label = torch.arange(sentence_num)[None, :].to(reorder_label.device)
        # true_label_matrix = torch.lt(true_label[:, :, None], true_label[:, None, :]).to(torch.float)
        # reorder_mask *= true_label_matrix
        # reorder_mask *= (1 - torch.lt(true_label[:, :, None] + 2, true_label[:, None, :]).to(torch.float))
        #
        # tmp_minus = reorder_label[:, None, :] - reorder_label[:, :, None]
        # reorder_label_matrix = (torch.lt(tmp_minus, 3) & torch.lt(-tmp_minus, 0)).to(torch.float)
        # batch_reorder_loss = -torch.log(pred_score + 1e-20) * reorder_label_matrix - \
        #                      torch.log(1 - pred_score + 1e-20) * (1 - reorder_label_matrix)
        # # deal with nan and inf
        # batch_reorder_loss_clean = torch.where(torch.isnan(batch_reorder_loss),
        #                                        torch.full_like(batch_reorder_loss, 0), batch_reorder_loss)
        # batch_reorder_loss_clean = torch.where(torch.isinf(batch_reorder_loss_clean),
        #                                  torch.full_like(batch_reorder_loss_clean, 10), batch_reorder_loss_clean)
        #
        # batch_reorder_loss_clean *= reorder_mask
        # reorder_loss = torch.mean(
        #     torch.sum(batch_reorder_loss_clean, [1, 2]) / (torch.sum(reorder_mask, [1, 2]) + 1e-20) * (
        #             1 - 0.99 * batch["normal_labels"]))

        # if hparams.sbert:
        try:
            sen_idx = sen_pos.nonzero()

            # [bath_size, sentence_num, sentence_num]
            sbert_score_label = batch["sbert_score"]
            sentence_num = sbert_score_label.size()[1]

            # [batch_size, sentence_num, hidden_size]
            sent_hidden_states_gather = self.gather_nd(hidden_states, sen_idx)
            sent_hidden_states = torch.reshape(sent_hidden_states_gather, [batch_size, sentence_num, hidden_size])
            self.sbert_linear_layer = self.sbert_linear_layer.float()

            # [batch_size, sentence_num, sentence_num]
            pred = torch.matmul(self.sbert_linear_layer(sent_hidden_states), torch.transpose(sent_hidden_states, 1, 2))
            pred_score = -1 + 2 * torch.sigmoid(pred + torch.transpose(pred, 1, 2))

            true_label = torch.arange(sentence_num)[None, :].to(pred.device)
            true_label_matrix = torch.le(true_label[:, :, None], true_label[:, None, :]).to(torch.float)
            sbert_mask = torch.ones_like(pred_score)

            batch_sbert_loss = torch.max(torch.abs(pred_score - sbert_score_label) - 0.1,
                                         torch.zeros_like(pred_score).to(pred_score.device))
            batch_sbert_loss *= sbert_mask
            sbert_loss = 0.1 * torch.sum(batch_sbert_loss) / (torch.sum(sbert_mask) + 1e-20)
        except Exception as e:
            # there is a bug in the test process
            if self.training:
                raise e
            else:
                print("error occured when calculating sbert_loss.")
                print(str(e))
                try:
                    print(f"the shape of sent_hidden_states_gather: {sent_hidden_states_gather.shape}")
                except Exception:
                    pass

        # reorder_loss has some problems. If we add it the loss would not downward.
        loss += sbert_loss
        return (loss, lm_loss, reorder_loss, sbert_loss)

    def training_step(self, batch, batch_idx) -> Dict:
        loss, lm_loss, reorder_loss, sbert_loss = self._step(batch)
        logs = {"loss": loss.item(), "lm_loss": lm_loss.item(),
                "reorder_loss": reorder_loss.item(), "sbert_loss": sbert_loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["tokens_per_batch"] = batch["input_ids"].ne(self.pad_token_id).sum() + batch["labels"].ne(
            self.pad_token_id).sum()
        logs["batch_size"] = batch["input_ids"].shape[0]
        logs["source_pad_tokens_num"] = batch["input_ids"].eq(self.pad_token_id).sum()
        logs["source_pad_tokens_ratio"] = batch["input_ids"].eq(self.pad_token_id).float().mean()
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        tik = datetime.now()
        if fast_generate:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )
        else:
            generated_ids = self.sample_sequence(batch, use_top_p=self.use_top_p, top_p=self.top_p)
        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(batch["labels"])
        loss, lm_loss, reorder_loss, sbert_loss = self._step(batch)

        base_metrics = {"loss": loss.item(), "lm_loss": lm_loss.item(),
                "reorder_loss": reorder_loss.item(), "sbert_loss": sbert_loss.item()}
        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> HINTLeadingContextSbertOrderDataset:
        datadir_name_ = Path(self.hparams.data_dir).name
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            hint_data_dir=f"{BASE_DIR}/datasets/thu-coai-hint/{datadir_name_}"
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

class EventHINT(LeadingContextHINT):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)
    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = HINTEventLineSbertOrderDataset

        self.reorder_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)
        self.sbert_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> HINTEventLineSbertOrderDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()

        datadir_name_ = Path(self.hparams.data_dir).name
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            hint_data_dir=f"{BASE_DIR}/datasets/thu-coai-hint/{datadir_name_}"
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

class LeadingPlusEventHINT(LeadingContextHINT):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = HINTLeadingPlusEventDataset
        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

        self.reorder_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)
        self.sbert_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> HINTLeadingPlusEventDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()

        datadir_name_ = Path(self.hparams.data_dir).name
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            event_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            hint_data_dir=f"{BASE_DIR}/datasets/thu-coai-hint/{datadir_name_}"
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
