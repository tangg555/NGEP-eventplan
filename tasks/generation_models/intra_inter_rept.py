"""
@Desc:
@Reference:
@Notes:
"""

import sys
from pathlib import Path
import argparse

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.generation_models.eval_utils import eval_intra_inter_repetitions
from src.utils import nlg_eval_utils
from src.utils.generation_models import visual_utils

# roc stories -------------------
roc_reference_intra = ({1: 0.0, 2: 1.28, 3: 1.72, 4: 2.03, 5: 2.18}, 1.8)
roc_reference_inter = ({1: 26.91, 2: 17.28, 3: 15.22, 4: 13.97}, 18.34)
# Seq2seq
roc_leading_Seq2seq_intra = ({1: 0.0, 2: 0.47, 3: 1.24, 4: 1.34, 5: 1.45}, 1.12)
roc_leading_Seq2seq_inter = ({1: 26.91, 2: 20.66, 3: 19.31, 4: 17.87}, 21.19)
roc_event_Seq2seq_intra = ({1: 0.0, 2: 0.25, 3: 1.25, 4: 1.55, 5: 1.52}, 1.14)
roc_event_Seq2seq_inter = ({1: 26.91, 2: 20.59, 3: 19.01, 4: 18.06}, 21.14)
roc_leading_plus_event_Seq2seq_intra = ({1: 0.0, 2: 0.51, 3: 1.25, 4: 1.52, 5: 1.5}, 1.2)
roc_leading_plus_event_Seq2seq_inter = ({1: 26.91, 2: 20.3, 3: 18.54, 4: 18.4}, 21.04)
# gpt2
roc_leading_gpt2_intra = ({1: 0.0, 2: 0.51, 3: 2.06, 4: 2.08, 5: 1.26}, 1.93)
roc_leading_gpt2_inter = ({1: 26.92, 2: 9.37, 3: 9.5, 4: 8.3}, 13.52)
roc_event_gpt2_intra = ({1: 0.0, 2: 0.43, 3: 1.98, 4: 3.4, 5: 4.4}, 2.55)
roc_event_gpt2_inter =  ({1: 26.91, 2: 29.85, 3: 28.8, 4: 26.84}, 28.1)
roc_leading_plus_event_gpt2_intra = ({1: 0.0, 2: 1.78, 3: 2.4, 4: 3.72, 5: 2.84}, 2.68)
roc_leading_plus_event_gpt2_inter = ({1: 26.9, 2: 22.22, 3: 22.02, 4: 22.9}, 23.51)
# bart
roc_leading_bart_intra = ({1: 0.0, 2: 2.11, 3: 1.77, 4: 1.86, 5: 1.97}, 1.93)
roc_leading_bart_inter = ({1: 26.91, 2: 19.6, 3: 17.65, 4: 17.08}, 20.31)
roc_event_bart_intra = ({1: 0.0, 2: 0.37, 3: 1.11, 4: 1.03, 5: 1.11}, 0.9)
roc_event_bart_inter = ({1: 26.92, 2: 17.12, 3: 15.58, 4: 16.5}, 19.03)
roc_leading_plus_event_bart_intra = ({1: 0.0, 2: 1.27, 3: 1.28, 4: 1.28, 5: 1.28}, 1.28)
roc_leading_plus_event_bart_inter = ({1: 26.91, 2: 16.32, 3: 15.68, 4: 14.32}, 18.31)
# hint
roc_leading_hint_intra = ({1: 0.0, 2: 1.76, 3: 1.68, 4: 1.88, 5: 1.92}, 1.81)
roc_leading_hint_inter = ({1: 26.91, 2: 20.44, 3: 18.27, 4: 17.23}, 20.71)
roc_event_hint_intra = ({1: 0.0, 2: 0.56, 3: 1.16, 4: 1.31, 5: 1.32}, 1.09)
roc_event_hint_inter = ({1: 26.91, 2: 20.79, 3: 21.28, 4: 14.88}, 20.96)
roc_leading_plus_event_hint_intra = ({1: 0.0, 2: 1.46, 3: 1.58, 4: 1.64, 5: 1.61}, 1.57)
roc_leading_plus_event_hint_inter = ({1: 26.91, 2: 18.36, 3: 15.98, 4: 14.33}, 18.89)
# etrica
roc_event_lm_sbert_intra = ({1: 0.0, 2: 1.15, 3: 1.2, 4: 1.25, 5: 1.21}, 1.2)
roc_event_lm_sbert_inter = ({1: 26.92, 2: 16.96, 3: 15.33, 4: 14.34}, 18.39)
roc_event_lm_intra = ({1: 0.0, 2: 1.41, 3: 1.53, 4: 1.72, 5: 1.71}, 1.59)
roc_event_lm_inter = ({1: 26.91, 2: 18.29, 3: 16.65, 4: 15.06}, 19.23)
roc_event_lm_sbert_no_cm_intra = ({1: 0.0, 2: 1.62, 3: 1.5, 4: 1.7, 5: 1.58}, 1.6)
roc_event_lm_sbert_no_cm_inter = ({1: 26.91, 2: 18.43, 3: 19.29, 4: 15.27}, 19.97)


def print_reference_repetition():
    ngram_counter = nlg_eval_utils.NGramCounter()
    for dataset_name in ["roc-stories", "writing-prompts"]:
        data_dir = Path(f"{BASE_DIR}/datasets/generation_models/{dataset_name}")
        src_file = data_dir.joinpath(f"test.source.txt")
        tgt_file = data_dir.joinpath(f"test.target.txt")
        src_lines = src_file.open("r", encoding="utf-8").readlines()
        tgt_lines = tgt_file.open("r", encoding="utf-8").readlines()
        assert len(src_lines) == len(tgt_lines)
        lines = [s_l + ' ' + t_l for s_l, t_l in zip(src_lines, tgt_lines)]
        sent_limit = 4 if dataset_name == "roc-stories" else 10
        intra_rept = ngram_counter.parse_lines_for_intra_repetition(lines, sent_limit=sent_limit+1, gram_n=2)
        inter_rept = ngram_counter.parse_lines_for_inter_repetition(lines, sent_limit=sent_limit, gram_n=3)
        print(f"{dataset_name} intra repetition:\n{intra_rept}")
        print(f"{dataset_name} inter repetition:\n{inter_rept}")
        print(f"p and d: {nlg_eval_utils.repetition_distinction_metric([line.strip().split() for line in lines])}")

def get_rept(dataset_name=None, model_name=None, file_prefix=None):
    # default args ---------
    dataset_name = "roc-stories" if dataset_name is None else dataset_name
    # default_dataset_name = "writing-prompts"
    model_name = "leading-bart" if model_name is None else model_name
    # file_prefix = "test_predicted_event"
    file_prefix = "test" if file_prefix is None else file_prefix
    sent_limit = 4
    # human input args ---------
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='roc-stories/writing-prompts')
    parser.add_argument('--gen_file', type=str,
                        default=f"output/generation_models/{model_name}-{dataset_name}/"
                                f"gen_result/{file_prefix}.source_gen.txt",
                        help='path of file containing generated text')
    args = parser.parse_args()

    data_dir = Path(f"{BASE_DIR}/datasets/generation_models/{args.dataset_name}")
    src_file = data_dir.joinpath(f"test.source.txt")
    # hint特殊，因为有相同的输入
    tgt_file = Path(f"{BASE_DIR}") / args.gen_file
    if not tgt_file.exists():
        raise FileNotFoundError(f"{tgt_file} does not exists")
    src_file = src_file if isinstance(src_file, Path) else Path(src_file)
    tgt_file = tgt_file if isinstance(tgt_file, Path) else Path(tgt_file)
    src_lines = src_file.open("r", encoding="utf-8").readlines()
    tgt_lines = tgt_file.open("r", encoding="utf-8").readlines()
    if "hint" in str(tgt_file.absolute()):
        new_tgt_lines = [tgt_lines[idx] for idx in range(0, len(tgt_lines), 2)]
        tgt_lines = new_tgt_lines
    assert len(src_lines) == len(tgt_lines)
    print(eval_intra_inter_repetitions(src_lines=src_lines, tgt_lines=tgt_lines,
                                       sent_limit=sent_limit, intra_gram_n=2, inter_gram_n=3))

def stat_plot_1(model_type="bart"):
    for metrics_type in ["intra", "inter"]:
        metrics_list = [eval(f"roc_leading_{model_type}_{metrics_type}"),
                              eval(f"roc_event_{model_type}_{metrics_type}"),
                              eval(f"roc_leading_plus_event_{model_type}_{metrics_type}")]
        visual_utils.rept_plot_for_comparison(metrics_list=metrics_list, title=f"{model_type} {metrics_type} repetition",
                                 figure=f"{model_type} {metrics_type} repetition", save_flag=False)
        visual_utils.aggre_rept_plot_for_comparison(metrics_list=metrics_list, title=f"{model_type} {metrics_type} repetition",
                                 figure=f"{model_type} {metrics_type} repetition", save_flag=False)

def stat_plot_2(model_type="bart", save_flag=False):
    metrics_type = "intra"
    intra_metrics_list = [eval(f"roc_leading_{model_type}_{metrics_type}"),
                          eval(f"roc_event_{model_type}_{metrics_type}"),
                          eval(f"roc_leading_plus_event_{model_type}_{metrics_type}")]
    metrics_type = "inter"
    inter_metrics_list = [eval(f"roc_leading_{model_type}_{metrics_type}"),
                          eval(f"roc_event_{model_type}_{metrics_type}"),
                          eval(f"roc_leading_plus_event_{model_type}_{metrics_type}")]
    visual_utils.complete_plot_for_comparison(intra_metrics_list=intra_metrics_list,
                                 inter_metrics_list=inter_metrics_list,
                                 model_type=model_type,
                                 title=f"{model_type} repetition",
                                 figure=f"{model_type} repetition", save_flag=save_flag)

# 总共4个图，分baseline 和 ablation来画
def stat_plot_3(save_flag=False):
    intra_metrics_dict = {"HINT": eval(f"roc_leading_plus_event_hint_intra"),
                          "BART": eval(f"roc_leading_plus_event_bart_intra"),
                          "EtriCA": eval(f"roc_event_lm_sbert_intra"),
                          "- w/o sen": eval(f"roc_event_lm_intra"),
                          "- w/o cm": eval(f"roc_event_lm_sbert_no_cm_intra"),
                          "- w/o leading": eval(f"roc_leading_hint_intra"),
                          "- w/o event": eval(f"roc_event_hint_intra"),
                          }

    visual_utils.plot_for_comparison_5_models(title="intra-rept",
                                 intra_metrics_dict=intra_metrics_dict,
                                 save_flag=save_flag)

def stat_plot_4(save_flag=False):
    intra_metrics_dict = {"HINT": eval(f"roc_leading_plus_event_hint_intra"),
                          "BART": eval(f"roc_leading_plus_event_bart_intra"),
                          "EtriCA": eval(f"roc_event_lm_sbert_intra"),
                          "- w/o sen": eval(f"roc_event_lm_intra"),
                          "- w/o cm": eval(f"roc_event_lm_sbert_no_cm_intra"),
                          "- w/o leading": eval(f"roc_leading_hint_intra"),
                          "- w/o event": eval(f"roc_event_hint_intra"),
                          }

    visual_utils.plot_for_comparison_5_models_together(title="intra-rept",
                                 intra_metrics_dict=intra_metrics_dict,
                                 save_flag=save_flag)

if __name__ == '__main__':
    # print_reference_repetition()
    # get_rept(dataset_name="roc-stories",
    #          model_name="event-hint",
    #          file_prefix="test_event")
    stat_plot_4(save_flag=True)



