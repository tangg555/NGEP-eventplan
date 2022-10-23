"""
@Desc:
@Reference:
- 坐标轴刻度间隔以及刻度范围
https://blog.csdn.net/weixin_44520259/article/details/89917026
@Notes:
"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.pyplot import MultipleLocator

from src.configuration.constants import BASE_DIR


def draw_rept_plot(metrics: dict, title: str = 'Repetition curve of reference text', figure="figure"):
    sent_ids: list = sorted(list(metrics.keys()))
    rept_scores: list = [metrics[idx] for idx in sent_ids]
    plt.figure(figure)
    plt.xticks(sent_ids)
    plt.plot(sent_ids, rept_scores, 'r--', label='reference text', linewidth=2.5, alpha=0.8)
    plt.plot(sent_ids, rept_scores, 'r*-', linewidth=2.5, alpha=0.8)
    plt.title(title)
    plt.xlabel('sent_idx')
    plt.ylabel('rept(%)')
    plt.legend()
    plt.show()


def rept_plot_for_comparison(metrics_list: list, title: str = 'Repetition curve of reference text', figure="figure",
                             save_dir=f"{BASE_DIR}/output/generation_models/cache/figs", save_flag=False):
    leading, event, lande = metrics_list
    sent_ids: list = sorted(list(leading[0].keys()))
    leading_rept_scores: list = [leading[0][idx] for idx in sent_ids]
    event_rept_scores: list = [event[0][idx] for idx in sent_ids]
    lande_rept_scores: list = [lande[0][idx] for idx in sent_ids]
    plt.figure(figure)
    plt.xticks(sent_ids)

    plt.plot(sent_ids, leading_rept_scores, 'r*-', label='leading', linewidth=2.5, alpha=0.8)
    plt.plot(sent_ids, event_rept_scores, 'g*-', label='event', linewidth=2.5, alpha=0.8)
    plt.plot(sent_ids, lande_rept_scores, 'b*-', label='leading+events', linewidth=2.5, alpha=0.8)
    plt.title(title)
    plt.xlabel('sent_idx')
    plt.ylabel('rept(%)')
    plt.legend()
    plt.show()
    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True)
        plt.savefig(save_dir / f"{title}.png")


def aggre_rept_plot_for_comparison(metrics_list: list, title: str = 'Repetition curve of reference text',
                                   figure="figure",
                                   save_dir=f"{BASE_DIR}/output/generation_models/cache/figs", save_flag=False):
    name_list = ["leading", "event", "leading+event"]
    aggre_scores = [one[1] for one in metrics_list]
    x = list(range(len(aggre_scores)))
    total_width, n = 0.8, 1
    width = total_width / n

    plt.figure(figure)

    plt.bar(x, aggre_scores, width=width, tick_label=name_list, fc='y')
    plt.title(title)
    plt.xlabel('sent_idx')
    plt.ylabel('rept(%)')
    plt.legend()
    plt.show()
    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True)
        plt.savefig(save_dir / f"{title}.png")


def complete_plot_for_comparison(intra_metrics_list: list, inter_metrics_list: list,
                                 model_type="bart",
                                 title: str = 'Repetition curve of reference text',
                                 figure="figure",
                                 save_dir=f"{BASE_DIR}/output/generation_models/cache/figs", save_flag=False):
    plt.figure(figure, figsize=(16, 4))
    name_list = ["leading", "event", "leading+event"]
    font_size = 12
    # intra rept -------------
    plt.subplot(1, 4, 1)
    # 设置`x`轴刻度间隔为`1`
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(labelsize=font_size)
    leading, event, lande = intra_metrics_list
    sent_ids: list = sorted(list(leading[0].keys()))
    leading_rept_scores: list = [leading[0][idx] for idx in sent_ids]
    event_rept_scores: list = [event[0][idx] for idx in sent_ids]
    lande_rept_scores: list = [lande[0][idx] for idx in sent_ids]
    plt.plot(sent_ids, leading_rept_scores, 'r*-', label='leading', linewidth=3, alpha=0.8)
    plt.plot(sent_ids, event_rept_scores, 'g*-', label='event', linewidth=3, alpha=0.8)
    plt.plot(sent_ids, lande_rept_scores, 'b*-', label='leading+events', linewidth=3, alpha=0.8)
    plt.xlabel('sent_idx')
    plt.ylabel('rept(%)')
    plt.title(f"{model_type} intra rep", fontsize=font_size)
    plt.legend()
    # intra aggre ------------
    plt.subplot(1, 4, 2)
    plt.tick_params(labelsize=font_size)
    aggre_scores = [one[1] for one in intra_metrics_list]
    x = list(range(len(aggre_scores)))
    width = 0.8
    plt.bar(x, aggre_scores, width=width, tick_label=name_list, color=["r", "g", "b"])
    plt.ylabel('rept(%)')
    interval = max(aggre_scores) - min(aggre_scores)
    plt.ylim(min(aggre_scores) - interval / 2.5, max(aggre_scores) + interval / 2.5)
    plt.title(f"{model_type} intra agg", fontsize=font_size)

    # inter rept ------------
    plt.subplot(1, 4, 3)
    # 设置`x`轴刻度间隔为`1`
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(labelsize=font_size)
    leading, event, lande = inter_metrics_list
    sent_ids: list = sorted(list(leading[0].keys()))
    leading_rept_scores: list = [leading[0][idx] for idx in sent_ids]
    event_rept_scores: list = [event[0][idx] for idx in sent_ids]
    lande_rept_scores: list = [lande[0][idx] for idx in sent_ids]
    plt.plot(sent_ids, leading_rept_scores, 'r*-', label='leading', linewidth=3, alpha=0.8)
    plt.plot(sent_ids, event_rept_scores, 'g*-', label='event', linewidth=3, alpha=0.8)
    plt.plot(sent_ids, lande_rept_scores, 'b*-', label='leading+events', linewidth=3, alpha=0.8)
    plt.xlabel('sent_idx')
    plt.ylabel('rept(%)')
    plt.title(f"{model_type} inter rep", fontsize=font_size)
    # inter aggre -----------
    plt.subplot(1, 4, 4)
    plt.tick_params(labelsize=font_size)
    aggre_scores = [one[1] for one in inter_metrics_list]
    x = list(range(len(aggre_scores)))
    width = 0.8
    plt.bar(x, aggre_scores, width=width, tick_label=name_list, color=["r", "g", "b"])
    plt.ylabel('rept(%)')
    interval = max(aggre_scores) - min(aggre_scores)
    plt.ylim(min(aggre_scores) - interval, max(aggre_scores) + interval)
    plt.title(f"{model_type} inter agg", fontsize=font_size)

    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{title}.png")
    plt.show()


def plot_for_comparison_5_models(intra_metrics_dict: dict,
                                 save_dir=f"{BASE_DIR}/output/generation_models/cache/figs", save_flag=False,
                                 title: str = "repetition for 5 models."):
    plt.figure("5_models_comparison", figsize=(16, 4))
    title = title
    font_size = 12
    # ===================== 1. =======================
    name_list = ["HINT",
                 "BART",
                 "EtriCA", ]
    intra_metrics_list = [intra_metrics_dict[name] for name in name_list]
    curve_colours = ['r*-', 'g*-', 'c*-']
    aggre_colours = ["r", "g", "c"]
    # intra rept -------------
    plt.subplot(1, 4, 1)
    # 设置`x`轴刻度间隔为`1`
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(labelsize=font_size)
    sent_ids: list = sorted(list(intra_metrics_list[0][0].keys()))
    intra_metrics_scores = []
    for model_metric in intra_metrics_list:
        intra_metrics_scores.append([model_metric[0][idx] for idx in sent_ids])

    for score, curve_color, name in zip(intra_metrics_scores, curve_colours, name_list):
        plt.plot(sent_ids, score, curve_color, label=name, linewidth=3, alpha=0.8)
    # plt.xlabel('(a) Intra-story repetition curve by sentence.')
    # plt.ylabel('rept(%)')
    # plt.title(f"Baseline models' comparison", fontsize=font_size)
    plt.ylim(0, 2.0)
    plt.legend()

    # intra aggre ------------
    plt.subplot(1, 4, 2)
    plt.tick_params(labelsize=font_size)
    aggre_scores = [one[1] for one in intra_metrics_list]
    x = list(range(len(aggre_scores)))
    width = 0.8
    plt.bar(x, aggre_scores, width=width, color=aggre_colours)
    # plt.xlabel('(b) Intra-story aggregate repetition scores.')
    # plt.ylabel('rept(%)')
    # plt.title(f"Baseline models' comparison", fontsize=font_size)
    interval = max(aggre_scores) - min(aggre_scores)
    plt.ylim(1.0, 1.8)

    # ===================== 2. =======================
    name_list = ["- w/o sen",
                 "- w/o cm",
                 "EtriCA",]
    intra_metrics_list = [intra_metrics_dict[name] for name in name_list]
    curve_colours = ['b*-', 'y*-', 'c*-']
    aggre_colours = [ "b", "y", "c"]
    # intra rept -------------
    plt.subplot(1, 4, 3)
    # 设置`x`轴刻度间隔为`1`
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(labelsize=font_size)
    sent_ids: list = sorted(list(intra_metrics_list[0][0].keys()))
    intra_metrics_scores = []
    for model_metric in intra_metrics_list:
        intra_metrics_scores.append([model_metric[0][idx] for idx in sent_ids])

    for score, curve_color, name in zip(intra_metrics_scores, curve_colours, name_list):
        plt.plot(sent_ids, score, curve_color, label=name, linewidth=3, alpha=0.8)
    # plt.xlabel('(c) Intra-story repetition curve by sentence.')
    # plt.ylabel('rept(%)')
    # plt.title(f"Ablation Study", fontsize=font_size)
    plt.ylim(0, 2.0)
    plt.legend()

    # intra aggre ------------
    plt.subplot(1, 4, 4)
    plt.tick_params(labelsize=font_size)
    aggre_scores = [one[1] for one in intra_metrics_list]
    x = list(range(len(aggre_scores)))
    width = 0.8
    plt.bar(x, aggre_scores, width=width, color=aggre_colours)
    # plt.xlabel('(d) Intra-story aggregate repetition scores.')
    # plt.ylabel('rept(%)')
    # plt.title(f"Ablation Study", fontsize=font_size)
    interval = max(aggre_scores) - min(aggre_scores)
    plt.ylim(1.0, 1.8)

    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{title}.svg")
    plt.show()

def plot_for_comparison_5_models_together(intra_metrics_dict: dict,
                                 save_dir=f"{BASE_DIR}/output/generation_models/cache/figs", save_flag=False,
                                 title: str = "repetition for 5 models."):
    plt.figure("5_models_comparison", figsize=(8, 4))
    title = title
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14,
            }
    # ===================== 1. =======================
    name_list = ["- w/o sen",
                 "- w/o cm",
                 "HINT",
                 "BART",
                 "EtriCA", ]
    intra_metrics_list = [intra_metrics_dict[name] for name in name_list]
    curve_colours = ['b*-', 'y*-', 'r*-', 'g*-', 'c*-']
    aggre_colours = ["b", "y", "r", "g", "c"]
    # intra rept -------------
    plt.subplot(1, 2, 1)
    # 设置`x`轴刻度间隔为`1`
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(font=font)
    plt.yticks(font=font)
    sent_ids: list = sorted(list(intra_metrics_list[0][0].keys()))
    intra_metrics_scores = []
    for model_metric in intra_metrics_list:
        intra_metrics_scores.append([model_metric[0][idx] for idx in sent_ids])

    for score, curve_color, name in zip(intra_metrics_scores, curve_colours, name_list):
        plt.plot(sent_ids, score, curve_color, label=name, linewidth=3, alpha=0.8)
    plt.ylim(0, 2.0)
    plt.legend()

    # intra aggre ------------
    plt.subplot(1, 2, 2)
    aggre_scores = [one[1] for one in intra_metrics_list]
    x = list(range(len(aggre_scores)))
    width = 0.8
    plt.xticks([], font=font)
    plt.yticks(font=font)
    plt.bar(x, aggre_scores, width=width, color=aggre_colours)
    plt.ylim(1.0, 1.8)

    if save_flag:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{title}.svg")
    plt.show()
if __name__ == '__main__':
    metrics = {1: 0.0, 2: 0.0016, 3: 0.0031, 4: 0.0042, 5: 0.0049}
    title = "intra repetition"
    draw_rept_plot(metrics, title, figure="intra repetition")
