"""
@Desc:
@Reference:
@Notes:
WANDB is Weights and Biases Logger:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
"""

import sys
import json
import numpy as np
from pathlib import Path
import torch

from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.generation_models.config_args import parse_args_for_config
from src.utils.model_utils import flatten_list
from src.utils.file_utils import pickle_load
from test import EventTriggerTester


class EventCaseTester(EventTriggerTester):
    def __init__(self, args):
        super().__init__(args)

    def instantly_generate(self, leading_context: str, event_line: str):
        batch = {}
        batch["leading_contexts"] = self.tokenizer(
            leading_context,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.args.max_target_length,
            return_tensors="pt",
        ).data
        batch["event_lines"] = self.tokenizer(
            event_line,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.args.max_target_length,
            return_tensors="pt",
        ).data
        generated_ids = self.model.sample_sequence(batch, use_top_p=True, top_p=0.9)
        generated_text = self.tokenizer.decode(flatten_list(generated_ids), skip_special_tokens=True)[0]
        print(f"leading_context: {leading_context} event_line: {event_line}")
        print(f"output: {generated_text}")
        return generated_text

    def case_study(self):
        print("golden event line")
        leading_context = "i needed a good plastic drink dispenser for halloween ."
        event_line = "[EVENT_s] gotten  [EVENT_sep] took her [EVENT_sep] was  [EVENT_sep] chose pink [EVENT_e]"
        print("predicted event line")
        self.instantly_generate(leading_context, event_line)
        predicted_event_line = "[EVENT_s] decided spend [EVENT_sep] shot himself [EVENT_sep] were  [EVENT_sep] was upset [EVENT_e]"
        self.instantly_generate(leading_context, predicted_event_line)
        print(f"referenced story: the pink dye had gotten all over them . her mother took her to get a new prescription ."
              f" it was time to order a new pair . she chose pink , and they both laughed at the irony . ")

    def batch_generate(self, batch: dict):
        generated_ids = self.model.sample_sequence(batch, use_top_p=True, top_p=0.9)
        generated_text = self.tokenizer.decode(flatten_list(generated_ids), skip_special_tokens=True)[0]
        leading_contexts = self.tokenizer.batch_decoder(batch["leading_contexts"], skip_special_tokens=True)
        event_lines = self.tokenizer.batch_decoder(batch["event_lines"], skip_special_tokens=True)
        print(f"leading_contexts: {leading_contexts}\nevent_lines: {event_lines}")
        print(f"output: {generated_text}")
        return generated_text


if __name__ == '__main__':
    # instant generation
    hparams = parse_args_for_config()
    hparams.model_name_or_path = f"{BASE_DIR}/output/generation_models/event-lm-roc-stories/best_tfmr"
    hparams.eval_batch_size = 1
    tester = EventCaseTester(hparams)
    tester.case_study()

