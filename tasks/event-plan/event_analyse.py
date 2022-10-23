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

from event_predictor import *



if __name__ == '__main__':
    from preprocessing.event_plan.event_extractor import EventExtractor

    dataset_name = "roc-stories"
    event_predictor = EventPredictor(dataset_name=dataset_name,
                                     event_extractor_path=f"{BASE_DIR}/output/event-plan/cache/{dataset_name}_event_graph.pkl",
                                     cache_dir=f"{BASE_DIR}/output/event-plan/cache",
                                     data_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}",
                                     output_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}")

    line = "[FEMALE] had a hard test in school she needed to study for ."
    leading_context = event_predictor.rm_extra_spaces(line)
    line_doc = event_predictor.nlp(leading_context)
    leading_event_list = []
    for sent_doc in line_doc.sents:
        event = event_predictor.extract_event_from_sent(sent_doc)
        leading_event_list.append(event)
    leading_event = leading_event_list[-1]
    event_list: list = event_predictor.frequency_based_inference(leading_context=leading_context,
                                                      leading_event=leading_event,
                                                      min_events=4,
                                                      max_events=4)
    print(event_list)


