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

from src.configuration.event_plan.config_args import parse_args_for_config
from src.utils.model_utils import flatten_list
from src.utils.file_utils import pickle_load
from predict import EventPlanPredictor


class EventPlanCaseTester(EventPlanPredictor):
    def __init__(self, args):
        super().__init__(args)

    def instantly_generate(self, input: str):
        batch = self.tokenizer(
            [input],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.args.max_target_length,
            return_tensors="pt",
        ).data
        batch["ids"] = [1]
        generated_ids = self.model.sample_sequence(batch, use_top_p=True, top_p=0.9)
        generated_text = self.tokenizer.decode(flatten_list(generated_ids), skip_special_tokens=True)
        return generated_text

    def case_study(self):
        print("=================== from test dataset ===================")
        leading_context = "i needed a good plastic drink dispenser for halloween ."
        event_line = "[EVENT_s] gotten  [EVENT_sep] took her [EVENT_sep] was  [EVENT_sep] chose pink [EVENT_e]"
        input = "i needed a good plastic drink dispenser for halloween . [EVENT_s] gotten [EVENT_sep]"
        print(f"leading context: {leading_context}")
        print(f"reference event_line: {event_line}")
        print(f"input: {input}")
        print(f"predicted event: {self.instantly_generate(input=input)}")
        print(f"referenced story: the pink dye had gotten all over them . her mother took her to get a new prescription ."
              f" it was time to order a new pair . she chose pink , and they both laughed at the irony . ")

        print("=================== from train dataset ===================")
        leading_context = "[MALE] 's parents were overweight ."
        event_line = "[EVENT_s] was overweight [EVENT_sep] told parents was [EVENT_sep] understood [EVENT_sep] got themselves [EVENT_e]"
        input = "[MALE] 's parents were overweight . [EVENT_s] was overweight [EVENT_sep]"
        print(f"leading context: {leading_context}")
        print(f"reference event_line: {event_line}")
        print(f"input: {input}")
        print(f"predicted event: {self.instantly_generate(input=input)}")
        print(f"referenced story: the pink dye had gotten all over them . her mother took her to get a new prescription ."
              f" it was time to order a new pair . she chose pink , and they both laughed at the irony . ")

if __name__ == '__main__':
    # instant generation
    hparams = parse_args_for_config()
    hparams.model_name_or_path = f"{BASE_DIR}/output/event-plan/event-plan-bart-roc-stories/best_tfmr"
    hparams.eval_batch_size = 1
    tester = EventPlanCaseTester(hparams)
    tester.case_study()

