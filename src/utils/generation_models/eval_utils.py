"""
@Desc:
@Reference:
@Notes:
"""

import os
import sys
from pathlib import Path

from collections import Counter
from typing import List

from src.configuration.constants import BASE_DIR
from src.utils import nlg_eval_utils

def eval_intra_inter_repetitions(src_lines:List[str], tgt_lines:List[str],
                                 sent_limit=4, intra_gram_n=2, inter_gram_n=3):
    ngram_counter = nlg_eval_utils.NGramCounter()
    assert len(src_lines) == len(tgt_lines)
    lines = [s_l + ' ' + t_l for s_l, t_l in zip(src_lines, tgt_lines)]
    intra_rept = ngram_counter.parse_lines_for_intra_repetition(lines, sent_limit=sent_limit+1, gram_n=intra_gram_n)
    inter_rept = ngram_counter.parse_lines_for_inter_repetition(lines, sent_limit=sent_limit, gram_n=inter_gram_n)
    return intra_rept, inter_rept

if __name__ == '__main__':
    pass