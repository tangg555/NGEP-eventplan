"""
@Desc:
@Reference:
@Notes:
- raw data
from thu-coai-hint roc-stories.
"""
import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.file_utils import copy_file_or_dir


def target_clean(tgt_file, out_file):
    flip = True
    with open(tgt_file, "r", encoding="utf-8") as fr, \
            open(out_file, "w", encoding="utf-8") as fw:
        for line in fr:
            flip = not flip
            if flip:
                continue
            fw.write(line.replace("<mask><s>", " "))

def src_clean(src_file, out_file):
    flip = True
    with open(src_file, "r", encoding="utf-8") as fr, \
            open(out_file, "w", encoding="utf-8") as fw:
        for line in fr:
            flip = not flip
            if flip:
                continue
            fw.write(line)

def read_src_and_tgt(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as src_fr, \
            open(tgt_file, "r", encoding="utf-8") as tgt_fr:
        return [sl.strip() + " " + tl.strip() for sl, tl in zip(src_fr, tgt_fr)]


def write_to_corpus(data_dir: Path, output_dir, corpus_file_name="all_data.txt", splits=None):
    if splits is None:
        splits = ["train", "val", "test"]
    # train and val to corpus
    with open(os.path.join(output_dir, corpus_file_name), "w", encoding="utf-8") as fw_corpus:
        lines = []
        for s_ in splits:
            lines += read_src_and_tgt(data_dir.joinpath(f"{s_}.source.txt"), data_dir.joinpath(f"{s_}.target.txt"))
        fw_corpus.write("\n".join(lines))
        print(f"data to {os.path.join(output_dir, corpus_file_name)}")


if __name__ == '__main__':
    src_dir = Path(f"{BASE_DIR}/resources/datasets/thu-coai-hint/roc-stories")
    output_dir = Path(f"{BASE_DIR}/resources/datasets/generation_models/roc-stories")
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix in ["train", "val", "test"]:
        src_clean(src_dir.joinpath(f"{prefix}.source"), output_dir.joinpath(f"{prefix}.source.txt"))
        target_clean(src_dir.joinpath(f"{prefix}.target"), output_dir.joinpath(f"{prefix}.target.txt"))
    write_to_corpus(output_dir, output_dir, corpus_file_name="corpus.txt", splits=["train", "val"])
    write_to_corpus(output_dir, output_dir, corpus_file_name="all_data.txt", splits=["train", "val", "test"])