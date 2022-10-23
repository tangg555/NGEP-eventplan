"""
@Desc:
@Reference:
@Notes:
"""
import json
import sys
import os
from shutil import copyfile
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils import nlg_eval_utils
from src.utils.generation_models.stat_utils import parse_files, parse_event_graphs
from preprocessing.generation_models.event_ontology import EventGraph
from src.utils.generation_models.event_analyzer_utils import EventAnalyzer
from src.utils.string_utils import rm_extra_spaces


def jaccard_score(string1: str, string2: str):
    seta = set(string1.strip().split())
    setb = set(string2.strip().split())
    jaccard_distance = lambda seta, setb: len(seta & setb) / float(len(seta | setb))
    score = jaccard_distance(seta, setb)
    return score


def events_similarity(pre_file: Path, ref_file: Path, lc_file: Path, story_file: Path):
    with pre_file.open("r", encoding="utf-8") as fr1, \
            ref_file.open("r", encoding="utf-8") as fr2:
        pre_lines = fr1.readlines()
        ref_lines = fr2.readlines()
        leading_contexts = lc_file.open("r", encoding="utf-8").readlines()
        stories = story_file.open("r", encoding="utf-8").readlines()
        scores = []
        idx = 0
        for pre, ref in tqdm(list(zip(pre_lines, ref_lines)), desc=f"pre_file: {pre_file}"):
            # scores.append((jaccard_score(pre.strip(), ref.strip()), idx))
            pre_set = set([one.strip() for one in pre.strip().split(EventGraph.event_sep)])
            ref_set = set([one.strip() for one in ref.strip().split(EventGraph.event_sep)])
            scores.append((len(pre_set & ref_set), idx))  # the number of same events
            idx += 1
        scores.sort(key=lambda x: x[0], reverse=True)

        most_high_ids = 10
        for idx in range(most_high_ids):
            score, score_idx = scores[idx]
            print(
                f"The {idx} highest||Story idx: {score_idx}||leading context:{leading_contexts[score_idx]}||stories:{stories[score_idx]}"
                f"||pre_line:{pre_lines[score_idx]}||ref_line:{ref_lines[score_idx]}||score:{score}")


def clean_line(line: str):
    line = line.strip()
    return rm_extra_spaces(line)


def events_metrics(pre_file: Path, ref_file: Path):
    with pre_file.open("r", encoding="utf-8") as fr1, \
            ref_file.open("r", encoding="utf-8") as fr2:
        pre_lines = [clean_line(one) for one in fr1.readlines()]
        ref_lines = [clean_line(one) for one in fr2.readlines()]

        pre_toks = [one.strip().split() for one in pre_lines]
        ref_toks = [one.strip().split() for one in ref_lines]
        metrics = {}
        # calculate bleu score
        nlg_eval_utils.calculate_bleu(ref_lines=pre_toks, gen_lines=ref_toks, metrics=metrics)
        # calculate rouge score
        rouge_metrics = nlg_eval_utils.calculate_rouge(pred_lines=pre_lines, tgt_lines=ref_lines)
        metrics.update(**rouge_metrics)
        # calculate repetition and distinction
        nlg_eval_utils.repetition_distinction_metric(pre_toks, metrics=metrics, repetition_times=2)
        print(f"pre_file: {pre_file}; metrics:\n{json.dumps(metrics, indent=4)}")


def retrieve_line(story_path: Path, story_idx: int):
    with story_path.open("r", encoding="utf-8") as fr:
        lines = fr.readlines()
        print(f"Path: {story_path.name}\n story_idx: {story_idx}; line: {lines[story_idx].strip()}")


if __name__ == '__main__':
    data_dir = Path(f"{BASE_DIR}/output/event-plan/storylines_set")
    predicted_file = data_dir.joinpath("test_predicted_event.source.txt")
    # predicted_file = data_dir.joinpath("leading-to-events-bart-roc-stories.test.source_gen.txt")
    reference_file = data_dir.joinpath("test_event.source.txt")
    leading_file = data_dir.joinpath("test.source.txt")
    story_file = data_dir.joinpath("test.target.txt")

    events_files = {"epb": data_dir.joinpath("event-plan-bart-roc-stories.event_plan_bart_event_gen.txt"),
                    "bart": data_dir.joinpath("leading-to-events-bart-roc-stories.test.source_gen.txt"),
                    "gpt2": data_dir.joinpath("leading-to-events-gpt2-roc-stories.test.source_gen.txt"),
                    "Seq2seq": data_dir.joinpath("leading-to-events-Seq2seq-roc-stories.test.source_gen.txt"),
                    "GNEP": data_dir.joinpath("test_predicted_event.source.txt")}

    # events_similarity(pre_file=predicted_file,
    #                   ref_file=reference_file,
    #                   lc_file=leading_file,
    #                   story_file=story_file)
    for name, event_file in events_files.items():
        retrieve_line(story_path=Path(event_file),
                      story_idx=302)

    # with reference_file.open("r", encoding="utf-8") as fr1:
    #     pre_lines = [clean_line(one) for one in fr1.readlines()]
    #     pre_toks = [one.strip().split() for one in pre_lines]
    #     print(f"ref events: {nlg_eval_utils.repetition_distinction_metric(pre_toks, repetition_times=2)}")

    # for name, event_file in events_files.items():
    #     print(f"============ {name} =============")
    #     events_metrics(pre_file=Path(event_file),
    #                    ref_file=reference_file)
