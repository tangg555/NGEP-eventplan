"""
@Desc:
@Reference:
@Notes:
- raw data
from thu-coai-hint writing-prompts. (remaining first 10 sentences)
"""
import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from preprocessing.event_plan.hint_roc_stories_helper import write_to_corpus
from hint_roc_stories_helper import src_clean, target_clean

def read_src_and_tgt(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as src_fr, \
            open(tgt_file, "r", encoding="utf-8") as tgt_fr:
        return [sl.strip() + " " + tl.strip() for sl, tl in zip(src_fr, tgt_fr)]


if __name__ == '__main__':
    src_dir = Path(f"{BASE_DIR}/resources/datasets/thu-coai-hint/writing-prompts")
    output_dir = Path(f"{BASE_DIR}/resources/datasets/generation_models/writing-prompts")
    output_dir.mkdir(exist_ok=True)

    for prefix in ["train", "val", "test"]:
        src_clean(src_dir.joinpath(f"{prefix}.source"), output_dir.joinpath(f"{prefix}.source.txt"))
        target_clean(src_dir.joinpath(f"{prefix}.target"), output_dir.joinpath(f"{prefix}.target.txt"))
    write_to_corpus(output_dir, output_dir, corpus_file_name="corpus.txt", splits=["train", "val"])
    write_to_corpus(output_dir, output_dir, corpus_file_name="all_data.txt", splits=["train", "val", "test"])