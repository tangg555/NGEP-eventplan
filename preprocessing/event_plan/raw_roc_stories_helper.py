"""
@Desc:
@Reference:
@Notes:
- raw data
from raw_data roc-stories
"""
import os
import sys

import spacy
from spacy.tokens.doc import Doc
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path


def split_roc_df(roc_df: pd.DataFrame, split_ratio: list = None):
    if split_ratio is None:
        split_ratio = [0.9, 0.05, 0.05]
    elif np.sum(split_ratio) == 1:
        split_ratio = split_ratio
    else:
        raise ValueError(f"invalid ratio: {split_ratio}")
    data_size = roc_df.shape[0]
    train_split = int(data_size * split_ratio[0])
    val_split = train_split + int(data_size * split_ratio[1])
    train_df, val_df, test_df = roc_df.iloc[:train_split], roc_df.iloc[train_split:val_split], roc_df.iloc[val_split:]
    return train_df, val_df, test_df

def write_df_to_files(data_df: pd.DataFrame, output_dir: str, file_type: str):
    leading_context = data_df['sentence1'].tolist()
    target_temp = data_df[["sentence2", "sentence3", "sentence4", "sentence5"]].values
    target = [" ".join(sent_list) for sent_list in target_temp]
    file_source = os.path.join(output_dir, f"{file_type}.source.txt")
    file_target = os.path.join(output_dir, f"{file_type}.target.txt")
    with open(file_source, 'w', encoding='utf-8') as fw_source, \
            open(file_target, 'w', encoding='utf-8') as fw_target:
        fw_source.write("\n".join(leading_context))
        fw_target.write("\n".join(target))


if __name__ == '__main__':
    data_path = Path(f"{BASE_DIR}/resources/raw_data/roc-stories/100KStories.csv")
    output_dir = Path(f"{BASE_DIR}/resources/datasets/generation_models/roc-stories")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["storyid", "storytitle", "sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]

    roc_df = pd.read_csv(data_path, sep=',')

    train_df, val_df, test_df = split_roc_df(roc_df, split_ratio=[0.9, 0.05, 0.05])
    print(f"文件大小: train-{train_df.shape[0]} val-{val_df.shape[0]} test-{test_df.shape[0]}")
    # write to files
    write_df_to_files(train_df, str(output_dir), "train")
    write_df_to_files(val_df, str(output_dir), "val")
    write_df_to_files(test_df, str(output_dir), "test")

    # train and val to corpus
    corpus_temp = pd.concat([train_df, val_df])[["sentence1", "sentence2", "sentence3",
                                                 "sentence4", "sentence5"]].values
    corpus = [" ".join(sent_list) for sent_list in corpus_temp]
    with open(os.path.join(output_dir, "train_val_corpus.txt"), "w", encoding="utf-8") as fw_corpus:
        fw_corpus.write("\n".join(corpus))

    # all data to corpus
    corpus_temp = pd.concat([train_df, val_df, test_df])[["sentence1", "sentence2", "sentence3",
                                                 "sentence4", "sentence5"]].values
    corpus = [" ".join(sent_list) for sent_list in corpus_temp]
    with open(os.path.join(output_dir, "corpus.txt"), "w", encoding="utf-8") as fw_corpus:
        fw_corpus.write("\n".join(corpus))
