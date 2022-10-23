"""
@Desc:
@Reference:
@Notes:
"""

import sys
import os
from shutil import copyfile
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.generation_models.stat_utils import parse_files, parse_event_graphs
from preprocessing.event_plan.event_predictor import EventPredictor
from src.utils.generation_models.event_analyzer_utils import EventAnalyzer


def print_stats():
    for dataset_name in ["roc-stories", "writing-prompts"]:
        data_dir = Path(f"{BASE_DIR}/datasets/generation_models/{dataset_name}")
        for prefix in ["train", "val", "test"]:
            stat_ = parse_files(src_file=data_dir.joinpath(f"{prefix}.source.txt"),
                                tgt_file=data_dir.joinpath(f"{prefix}.target.txt"),
                                event_file=data_dir.joinpath(f"{prefix}_event.source.txt"))
            stat_.update(parse_event_graphs(dataset_name))
            print(f"{prefix}: {stat_}")


def load_event_predictor(dataset_name) -> EventPredictor:
    predictor = EventPredictor(name=dataset_name,
                               event_extractor_path=f"{BASE_DIR}/output/generation_models/cache/{dataset_name}_event_graph.pkl",
                               cache_dir=f"{BASE_DIR}/output/generation_models/cache",
                               data_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}",
                               output_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}")
    return predictor

def are_all_key_words_contained(key_words:list, string:str):
    for key in key_words:
        if key not in string:
            return False
    return True

def is_any_key_words_contained(key_words:list, string:str):
    for key in key_words:
        if key in string:
            return True
    return False

def retrieve_gen_result(dir_key_words:list,
                        file_key_words:list=None,
                        discard_dir_key_words: list = None,
                        discard_file_key_words:list=None,
                        result_dir=f'{BASE_DIR}/output/generation_models',
                        output_dir=f'{BASE_DIR}/output/generation_models/gen_result',
                        file_name_key_words:list=None):
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    if not result_dir.exists():
        raise FileNotFoundError()
    output_dir.mkdir(parents=True, exist_ok=True)
    tgt_src_files = []
    tgt_out_files = []
    for subdir_name in os.listdir(result_dir):
        # subdir_name contains all those key_words
        if are_all_key_words_contained(dir_key_words, subdir_name):
            # skip if contain these key words
            if discard_dir_key_words and is_any_key_words_contained(discard_dir_key_words, subdir_name):
                continue  # skip
            for file_name in os.listdir(result_dir / subdir_name / "gen_result"):
                    # only keep files which contain the file_name_key_words
                    if file_key_words and not are_all_key_words_contained(file_key_words, file_name):
                        continue # skip
                    # only keep files which don't contain the discard_file_key_words
                    if discard_file_key_words and is_any_key_words_contained(discard_file_key_words, file_name):
                        continue  # skip
                    tgt_src_files.append(result_dir / subdir_name / "gen_result" / file_name)
                    tgt_out_files.append(output_dir / f"{subdir_name}.{file_name}")

    for src_path, out_path in zip(tgt_src_files, tgt_out_files):
        copyfile(src_path, out_path)

def clean_hint_gen_file(gen_file):
    print(f"clean_hint_gen_file: {gen_file}")
    with open(gen_file, "r", encoding="utf-8") as fr:
        # firstly read and then write
        pre_lines = fr.readlines()
        new_lines = []
        for i in range(0, len(pre_lines), 2):
            new_lines.append(pre_lines[i])
    with open(gen_file, "w", encoding="utf-8") as fw:
        fw.writelines(new_lines)
        print(f"prev_lines:{len(pre_lines)}, new_lines:{len(new_lines)}")

if __name__ == '__main__':
    from preprocessing.event_plan.event_extractor import EventExtractor  # cannot be removed

    # for etrica ===============
    discard_dir_key_words = ["gpt", "Seq2seq"]
    discard_file_key_words = ["predicted", "bart"]
    output_dir = f'{BASE_DIR}/output/generation_models/gen_result_for_etrica'
    # roc stories
    retrieve_gen_result(dir_key_words=["leading-plus-event", "roc-stories"],
                        file_key_words=["gen.txt"],
                        discard_dir_key_words=discard_dir_key_words,
                        discard_file_key_words=discard_file_key_words,
                        result_dir=f'{BASE_DIR}/output/generation_models',
                        output_dir=output_dir)
    retrieve_gen_result(dir_key_words=["event-lm", "roc-stories"],
                        file_key_words=["gen.txt"],
                        discard_dir_key_words=discard_dir_key_words,
                        discard_file_key_words=discard_file_key_words,
                        result_dir=f'{BASE_DIR}/output/generation_models',
                        output_dir=output_dir)
    clean_hint_gen_file(f'{output_dir}/leading-plus-event-hint-roc-stories.test_event.source_gen.txt')