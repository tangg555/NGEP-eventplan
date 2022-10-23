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
from typing import List
from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.event_plan.config_args import parse_args_for_config
from src.utils.file_utils import copy_file_or_dir, output_obj_to_file, pickle_save, pickle_load
from src.utils import nlg_eval_utils
from src.utils.model_utils import flatten_list
from src.utils.string_utils import rm_extra_spaces
from preprocessing.event_plan.event_ontology import EventGraph
from train import EventPlanTrainer

class EventPlanPredictor(EventPlanTrainer):
    def __init__(self, args):
        # parameters
        super().__init__(args)
        self.generation_dir = self.experiment_output_dir / "gen_result"
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = self.model.tokenizer
        self.model.eval()
        self.test_output = {}
        self.leading_file = None
        self.event_file = None
        self.gen_file = None
        self.eval_file = None
        self.ppl_file = None

        # customized
        self.dataset = self.model.test_dataloader().dataset
        self.leading_file = self.dataset.leading_file
        self.event_file = self.dataset.event_file
        self.output_prefix = f"event_plan_bart_event"
        self.test_output_store_path = self.cache_dir.joinpath(f"{self.output_prefix}_test_output.pkl")
        self.gen_file = self.generation_dir / f"{self.output_prefix}_gen.txt"
        self.eval_file = self.generation_dir / f"{self.output_prefix}_eval.txt"

        self.leading_data = self._read_clean_lines(self.leading_file)
        self.event_data = self._read_clean_lines(self.event_file)

    def _read_clean_lines(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = rm_extra_spaces(line)
                if len(line) > 0:
                    data.append(line)
        return data

    def batch_generate(self, leadings: List[str], ids: List[int]):
        batch = self.tokenizer(
            leadings,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.args.max_target_length,
            return_tensors="pt",
        ).data
        batch["ids"] = ids
        generated_ids = self.model.sample_sequence(batch, use_top_p=True, top_p=0.9)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def generate(self):
        if self.gen_file.exists():
            with open(self.gen_file, "r", encoding="utf-8") as fw_in:
                self.test_output["preds"] = [one.strip() for one in fw_in.readlines()]
        else:
            # encode数据并sample
            data_size = len(self.leading_data)
            batch_size = self.args.eval_batch_size
            event_nums = 4
            all_generated_texts = []
            for start_idx in tqdm(list(range(0, data_size, batch_size)), desc="generate events..."):
                end_idx = start_idx + batch_size if start_idx + batch_size <= data_size else data_size
                inputs = [f"{one.strip()} {EventGraph.event_s}" for one in self.leading_data[start_idx: end_idx]]
                # 拼接 event_nums次
                for count in range(event_nums):
                    generated_texts = self.batch_generate(inputs, list(range(start_idx, end_idx)))
                    assert len(inputs) == len(generated_texts)
                    if count == event_nums-1:
                        new_ = [f"{input.strip()} {next_event} {EventGraph.event_e} "
                                for input, next_event in zip(inputs, generated_texts)]
                    else:
                        new_ = [f"{input.strip()} {next_event} {EventGraph.event_sep} "
                                for input, next_event in zip(inputs, generated_texts)]
                    inputs = new_
                # 只截取events的部分
                outputs = [input[input.find(f"{EventGraph.event_s}"):] for input in inputs]
                all_generated_texts.extend(outputs)
            assert len(all_generated_texts) == data_size
            print(f"model {self.model.model_name} generating")
            print(f"src_file: {self.leading_file}\ntgt_file: {self.event_file}\ngen_file: {self.gen_file}\n")
            self.test_output["preds"] = all_generated_texts

        self.test_output["tgts"] = [one.strip() for one in self.event_data]
        with open(self.gen_file, "w", encoding="utf-8") as fw_out:
            fw_out.write("\n".join(self.test_output["preds"]))

    def eval_output(self):
        pred_lines = self.test_output["preds"]
        tgt_lines = self.test_output["tgts"]
        tgt_lines_toks, pred_lines_toks = \
            [self.tokenizer.tokenize(t) for t in tgt_lines], [self.tokenizer.tokenize(c) for c in pred_lines]

        metrics = {}
        # calculate perplexity
        # calculate bleu score
        nlg_eval_utils.calculate_bleu(ref_lines=tgt_lines_toks, gen_lines=pred_lines_toks, metrics=metrics)
        # calculate rouge score
        rouge_metrics = nlg_eval_utils.calculate_rouge(pred_lines=pred_lines, tgt_lines=tgt_lines)
        metrics.update(**rouge_metrics)
        # calculate repetition and distinction
        nlg_eval_utils.repetition_distinction_metric(pred_lines_toks, metrics=metrics, repetition_times=2)
        key = sorted(metrics.keys())
        for k in key:
            print(k, metrics[k])
        print("=" * 10)

        print(f"model {self.model.model_name} eval {self.gen_file}")
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)
        return metrics

if __name__ == '__main__':
    hparams = parse_args_for_config()
    hparams.model_name_or_path = f"{BASE_DIR}/output/event-plan/event-plan-bart-roc-stories/best_tfmr"
    predictor = EventPlanPredictor(hparams)

    # generate predicted stories
    predictor.generate()
    predictor.eval_output()