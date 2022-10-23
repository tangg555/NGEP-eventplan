"""
@Desc:
@Reference:
@Notes:
"""
import json
import pickle
import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.generation_models.eval_utils import eval_intra_inter_repetitions
from src.utils import nlg_eval_utils
from src.utils.generation_models import visual_utils

# roc stories -------------------
roc_reference_intra = ({1: 0.0, 2: 1.28, 3: 1.72, 4: 2.03, 5: 2.18}, 1.8)
roc_reference_inter = ({1: 26.91, 2: 17.28, 3: 15.22, 4: 13.97}, 18.34)

REPT_CACHE_FILE = Path(f"{BASE_DIR}/output/event-plan/cache/rept_cache.pkl")
ROC_DATA_DIR = Path(f"{BASE_DIR}/output/event-plan/gen_results_summary")
ROC_STORY_CORPORA = {
    # Seq2Seq -----
    "Seq2seq-noevents": ROC_DATA_DIR.joinpath("leading-Seq2seq-roc-stories.test.source_gen.txt"),
    "Seq2seq-Seq2seq": ROC_DATA_DIR.joinpath(
        "leading-plus-event-Seq2seq-roc-stories.test_Seq2seq_event.source_gen.txt"),
    "Seq2seq-bart": ROC_DATA_DIR.joinpath(
        "leading-plus-event-Seq2seq-roc-stories.test_bart_event.source_gen.txt"),
    "Seq2seq-gpt2": ROC_DATA_DIR.joinpath(
        "leading-plus-event-Seq2seq-roc-stories.test_gpt2_event.source_gen.txt"),
    "Seq2seq-epb": ROC_DATA_DIR.joinpath(
        "leading-plus-event-Seq2seq-roc-stories.test_epb_event.source_gen.txt"),
    "Seq2seq-ngp": ROC_DATA_DIR.joinpath(
        "leading-plus-event-Seq2seq-roc-stories.test_predicted_event.source_gen.txt"),
    # BART -----
    "bart-noevents": ROC_DATA_DIR.joinpath("leading-bart-roc-stories.test.source_gen.txt"),
    "bart-Seq2seq": ROC_DATA_DIR.joinpath(
        "leading-plus-event-bart-roc-stories.test_Seq2seq_event.source_gen.txt"),
    "bart-bart": ROC_DATA_DIR.joinpath(
        "leading-plus-event-bart-roc-stories.test_bart_event.source_gen.txt"),
    "bart-gpt2": ROC_DATA_DIR.joinpath(
        "leading-plus-event-bart-roc-stories.test_gpt2_event.source_gen.txt"),
    "bart-epb": ROC_DATA_DIR.joinpath(
        "leading-plus-event-bart-roc-stories.test_epb_event.source_gen.txt"),
    "bart-ngp": ROC_DATA_DIR.joinpath(
        "leading-plus-event-bart-roc-stories.test_predicted_event.source_gen.txt"),
    # HINT -----
    "hint-noevents": ROC_DATA_DIR.joinpath("leading-hint-roc-stories.test.source_gen.txt"),
    "hint-Seq2seq": ROC_DATA_DIR.joinpath(
        "leading-plus-event-hint-roc-stories.test_Seq2seq_event.source_gen.txt"),
    "hint-bart": ROC_DATA_DIR.joinpath(
        "leading-plus-event-hint-roc-stories.test_bart_event.source_gen.txt"),
    "hint-gpt2": ROC_DATA_DIR.joinpath(
        "leading-plus-event-hint-roc-stories.test_gpt2_event.source_gen.txt"),
    "hint-epb": ROC_DATA_DIR.joinpath(
        "leading-plus-event-hint-roc-stories.test_epb_event.source_gen.txt"),
    "hint-ngp": ROC_DATA_DIR.joinpath(
        "leading-plus-event-hint-roc-stories.test_predicted_event.source_gen.txt"),
    # T5 -----
    "t5-noevents": ROC_DATA_DIR.joinpath("leading-t5-roc-stories.test.source_gen.txt"),
    "t5-Seq2seq": ROC_DATA_DIR.joinpath(
        "leading-plus-event-t5-roc-stories.test_Seq2seq_event.source_gen.txt"),
    "t5-bart": ROC_DATA_DIR.joinpath(
        "leading-plus-event-t5-roc-stories.test_bart_event.source_gen.txt"),
    "t5-gpt2": ROC_DATA_DIR.joinpath(
        "leading-plus-event-t5-roc-stories.test_gpt2_event.source_gen.txt"),
    "t5-epb": ROC_DATA_DIR.joinpath(
        "leading-plus-event-t5-roc-stories.test_epb_event.source_gen.txt"),
    "t5-ngp": ROC_DATA_DIR.joinpath(
        "leading-plus-event-t5-roc-stories.test_predicted_event.source_gen.txt"),
}


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
        intra_rept = ngram_counter.parse_lines_for_intra_repetition(lines, sent_limit=sent_limit + 1, gram_n=2)
        inter_rept = ngram_counter.parse_lines_for_inter_repetition(lines, sent_limit=sent_limit, gram_n=3)
        print(f"{dataset_name} intra repetition:\n{intra_rept}")
        print(f"{dataset_name} inter repetition:\n{inter_rept}")
        print(f"p and d: {nlg_eval_utils.repetition_distinction_metric([line.strip().split() for line in lines])}")


def get_rept(data_dir=None, gen_file=None):
    # default args ---------
    data_dir = ROC_DATA_DIR if data_dir is None else data_dir
    gen_file = "leading-bart" if gen_file is None else gen_file

    sent_limit = 4
    # human input args ---------
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_file', type=str,
                        default=f"{gen_file}",
                        help='path of file containing generated text')
    args = parser.parse_args()

    src_file = data_dir.joinpath(f"test.source.txt")
    # hint特殊，因为有相同的输入
    tgt_file = Path(args.gen_file)
    if not tgt_file.exists():
        raise FileNotFoundError(f"{tgt_file} does not exists")
    src_file = src_file if isinstance(src_file, Path) else Path(src_file)
    tgt_file = tgt_file if isinstance(tgt_file, Path) else Path(tgt_file)
    src_lines = src_file.open("r", encoding="utf-8").readlines()
    tgt_lines = tgt_file.open("r", encoding="utf-8").readlines()
    assert len(src_lines) == len(tgt_lines)
    rept_result = eval_intra_inter_repetitions(src_lines=src_lines, tgt_lines=tgt_lines,
                                               sent_limit=sent_limit, intra_gram_n=2, inter_gram_n=3)
    return rept_result


def store_rept_to_cache():
    all_results = {}
    for model, path_ in ROC_STORY_CORPORA.items():
        rept_result = get_rept(data_dir=ROC_DATA_DIR, gen_file=path_)
        print(f"{model} intra-rept: {rept_result[0]}; inter-rept: {rept_result[1]}")
        all_results[model] = rept_result
    pickle.dump(all_results, REPT_CACHE_FILE.open("wb"))


def load_rept_from_cache():
    return pickle.load(REPT_CACHE_FILE.open("rb"))


def retrieve_intra_metrics_by_model_name(model_name: str, metrics_dict: dict):
    expected_metrics = {}
    for name, metrics in metrics_dict.items():
        if f"{model_name}-" in name:
            expected_metrics[name[len(f"{model_name}-"):]] = metrics_dict[name][0][0]
    return expected_metrics


def draw_single_plot_for_plan(intra_metrics_dict: dict,
                              save_dir=f"{BASE_DIR}/output/event-plan/cache/figs", save_flag=False,
                              title: str = "Intra-story repetitions."):
    plt.figure("eventplan_comparison", figsize=(16, 4))
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14,
            }
    eventplan_name_map = {
        "w/o events": "noevents",
        "Seq2Seq": "Seq2seq",
        "BART": "bart",
        "GPT-2": "gpt2",
        "EventAdvisor": "epb",
        "NGP": "ngp"
    }
    curve_colours = ['m*-', 'k*-', 'b*-', 'y*-', 'r*-', 'g*-', 'c*-']

    # ===================== 1. =======================
    for graph_idx, model_name in enumerate(["Seq2Seq", "BART", "HINT", "T-5"]):
        name_list = ["w/o events",
                     "Seq2Seq",
                     "BART",
                     "GPT-2",
                     "EventAdvisor",
                     "NGP"]
        intra_metrics_list = [intra_metrics_dict[model_name][eventplan_name_map[name]]
                              for name in name_list]

        # intra rept -------------
        plt.subplot(1, 4, graph_idx + 1)
        # 设置`x`轴刻度间隔为`1`
        ax = plt.gca()
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xticks(font=font)
        plt.yticks(font=font)
        sent_ids: list = sorted(list(intra_metrics_list[0].keys()))
        intra_metrics_scores = []
        for model_metric in intra_metrics_list:
            intra_metrics_scores.append([model_metric[idx] for idx in sent_ids])

        for score, curve_color, name in zip(intra_metrics_scores, curve_colours, name_list):
            plt.plot(sent_ids, score, curve_color, label=name, linewidth=3, alpha=0.8)
        plt.ylim(0, 3)
        if graph_idx == 0:
            plt.legend(loc="upper left", prop={'family': 'Times New Roman',
                             'weight': 'bold',
                             'size': 10,
                             })

    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{title}.svg")
    plt.show()


def stat_plot_1(save_flag=False):
    all_result = load_rept_from_cache()
    intra_metrics_dict = {"Seq2Seq": retrieve_intra_metrics_by_model_name("Seq2seq", all_result),
                          "BART": retrieve_intra_metrics_by_model_name("bart", all_result),
                          "HINT": retrieve_intra_metrics_by_model_name("hint", all_result),
                          "T-5": retrieve_intra_metrics_by_model_name("t5", all_result),
                          }

    draw_single_plot_for_plan(title="intra-rept-for-eventplan",
                              intra_metrics_dict=intra_metrics_dict,
                              save_flag=save_flag)


if __name__ == '__main__':
    # print_reference_repetition()
    # store_rept_to_cache()
    stat_plot_1(save_flag=True)
