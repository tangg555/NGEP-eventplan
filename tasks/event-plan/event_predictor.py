"""
@Desc:
@Reference:
@Notes:
- 关于event planning.
有特殊性，并不是预测的跟原先序列一样的越好。
纯概率方法生成
待比较：
A Graph-to-Sequence Model for AMR-to-Text Generation
plan and write
"""

import sys
import os
from multiprocessing import Process
from pathlib import Path
from collections import Counter
from typing import List
import spacy
from tqdm import tqdm
import numpy as np

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.event_plan.config_args import parse_args_for_config
from preprocessing.event_plan.event_extractor import EventExtractor
from preprocessing.event_plan.event_ontology import Event, EventGraph

from case_test import EventPlanCaseTester

class EventPredictor(EventExtractor):
    def __init__(self, dataset_name: str = "roc",
                 event_extractor_path=f"{BASE_DIR}/output/event-plan/cache/dataset_name_event_graph.pkl",
                 data_dir=f"{BASE_DIR}/resources/raw_data/dataset_name",
                 output_dir=f"{BASE_DIR}/resources/datasets/event-plan/dataset_name",
                 cache_dir=f"{BASE_DIR}/output/event-plan/cache",
                 nlp: spacy.language.Language = None):
        super().__init__(dataset_name=dataset_name, data_dir=data_dir, output_dir=output_dir, cache_dir=cache_dir, nlp=nlp)
        self.event_graph = None
        if event_extractor_path is not None:
            event_extractor: EventExtractor = self.load(event_extractor_path)
            self.event_graph = event_extractor.event_graph
        # set random seed
        np.random.seed(42)
        self.store_path = self.cache_dir.joinpath(f"{self.dataset_name}_predictor.pkl")

        # stat
        self.predicted_empty_line_counter = Counter()
        self.predicted_lines = Counter()

        # load the neural event predict model -------
        hparams = parse_args_for_config()
        hparams.model_name_or_path = f"{BASE_DIR}/output/event-plan/event-plan-bart-{self.dataset_name}/best_tfmr"
        hparams.eval_batch_size = 1
        self.neural_planner = EventPlanCaseTester(hparams)

    def jaccard_score(self, string1: str, string2: str):
        seta = set(string1.strip().split())
        setb = set(string2.strip().split())
        jaccard_distance = lambda seta, setb: len(seta & setb) / float(len(seta | setb))
        score = jaccard_distance(seta, setb)
        return score

    def choose_closest_event_by_name(self, name, events: List[Event]):
        # similarity between text
        scores = [self.jaccard_score(name, one.string) for one in events]
        max_idx = np.argmax(scores)
        return events[max_idx]

    def let_neural_model_suggest(self, input_str: str, events: List[Event]) -> Event:
        event_candidate_name = self.neural_planner.instantly_generate(input=input_str).strip()
        return self.choose_closest_event_by_name(event_candidate_name, events)

    # The main algorithm to predict events.
    def frequency_based_inference(self,
                                  leading_context: str,
                                  leading_event: Event,
                                  min_events: int,
                                  max_events: int,
                                  max_con_rept=2):
        event_list = []
        event_counter = Counter()
        lead_event_id = leading_event.uuid
        all_available_events: List[Event] = list(self.event_graph.events.values())
        # continuously repeat events
        con_rept_count = 0

        # ------------- 1. decide the leading event ------------------
        # lead_event not in event_graph
        if lead_event_id not in self.event_graph.next_events:
            # find event whose string is the trigger ------
            leading_event_ = None
            for event_ in self.event_graph.events.values():
                if event_.string.strip() == leading_event.generation_models.strip():
                    leading_event_ = event_
            lead_event_id = None if leading_event_ is None else leading_event_.uuid

        # ------------- 2. infer the next event one by one ------------------
        next_event_id = lead_event_id
        event_count = 0
        while True:
            # leading event is not recorded in the eventgraph
            if next_event_id is None:
                next_event_temp = self.let_neural_model_suggest(f"{leading_context} {EventGraph.event_s}",
                                                               all_available_events)
                next_event_id_temp = next_event_temp.uuid
                next_event_id = next_event_id_temp
                event_list.append(self.event_graph.events[next_event_id].string)
                event_counter[next_event_id] += 1

                # complete
                event_count += 1
                if event_count >= max_events:
                    break
                continue

            candidate_counter_: Counter = self.event_graph.next_events[next_event_id].copy()
            # less than min_events: remove the end tag and then select
            if self.event_graph.event_e in candidate_counter_ and event_count < min_events:
                del candidate_counter_[self.event_graph.event_e]

            # select candidates for the next event
            candidates_ = list(candidate_counter_.keys())
            freq_ = list(candidate_counter_.values())  # frequency
            # punish the repetition
            repeti_punishment_factors_ = []
            for candi_uuid in candidates_:
                in_degree_of_candi = sum(list(self.event_graph.prev_events[candi_uuid].values()))
                factor = 1 if candi_uuid not in event_counter else \
                    float((1 / in_degree_of_candi) *
                          (max_con_rept-event_counter[candi_uuid])/max_con_rept)
                repeti_punishment_factors_.append(factor)
            freq_ = freq_ * np.array(repeti_punishment_factors_, dtype=float)
            # get the probability distribution for the event candidates
            probs_ = np.array(freq_, dtype=float) / np.sum(freq_)

            if len(candidates_) > 0:    # if there is at least 1 candidate
                next_event_id_temp = np.random.choice(candidates_, size=1, replace=False, p=probs_)[0]
            else:   # if there is no candidate to choose
                next_event_temp = self.let_neural_model_suggest(f"{leading_context} {EventGraph.event_s} "
                                                                f"{EventGraph.event_sep.join(event_list)}",
                                                               all_available_events)
                next_event_id_temp = next_event_temp.uuid

            # count events continuously repeated
            if next_event_id == next_event_id_temp:
                con_rept_count += 1

            # state choice
            if next_event_id_temp == self.event_graph.event_e: # it is the end of event plan
                break
            elif con_rept_count > max_con_rept: # it repeat too many times
                # if continuously repeated too much times then randomly choose another event
                next_event_ = self.let_neural_model_suggest(f"{leading_context} {EventGraph.event_s} "
                                                                f"{EventGraph.event_sep.join(event_list)}",
                                                               all_available_events)
                next_event_id = next_event_.uuid
                con_rept_count = 0  if next_event_id != next_event_id else con_rept_count
                continue
            else: # it is Ok to choose next_event_id_temp
                next_event_id = next_event_id_temp
                event_list.append(self.event_graph.events[next_event_id].string)
                event_counter[next_event_id] += 1

                # complete
                event_count += 1
                if event_count >= max_events:
                    break
        return event_list


    def predict_for_file(self, input_file="test.source.txt",
                         output_file="test_predicted_event.source.txt",
                         min_events=4,
                         max_events=20):
        input_path = self.data_dir.joinpath(input_file)
        output_path = self.output_dir.joinpath(output_file)
        with open(input_path, "r", encoding="utf-8") as fr, \
                open(output_path, "a+", encoding="utf-8") as fa:
            input_lines = [line.strip() for line in fr.readlines()]
            input_size = len(input_lines)
            fa.seek(0)  # read from line 0
            existing_size = len(fa.readlines())
            rest_size = input_size - existing_size
            if input_size < existing_size:
                raise ValueError(f"input_size: {input_size} < existing_size: {existing_size}")
            elif input_size == existing_size:
                return
            else:
                pass
            print(f"predicting file: {input_file}, total: {input_size}, "
                  f"already finished: {existing_size}, rest: {input_size - existing_size}")

            # predicting leading_context can has
            for line in tqdm(input_lines[existing_size:],
                                 total=rest_size,
                                 desc=f"predicting events for {input_file}, and output to {output_file}"):
                leading_context = self.rm_extra_spaces(line)
                line_doc = self.nlp(leading_context)
                leading_event_list = []
                for sent_doc in line_doc.sents:
                    event = self.extract_event_from_sent(sent_doc)
                    leading_event_list.append(event)
                leading_event = leading_event_list[-1]
                event_list: list = self.frequency_based_inference(leading_context=leading_context,
                                                                  leading_event=leading_event,
                                                                  min_events=min_events,
                                                                  max_events=max_events)
                if len(event_list) == 0:
                    self.predicted_empty_line_counter[output_file] += 1
                event_line = self.event_list_to_line(event_list)
                fa.write(event_line + "\n")
                self.predicted_lines[output_file] += 1

        print(f"empty lines:")
        for file_name in self.predicted_lines.keys():
            print(f"{file_name} total_predicted_lines: {self.predicted_lines[file_name]}, "
                  f"empty_lines: {self.predicted_empty_line_counter[file_name]}, "
                  f"ratio:"
                  f"{round(self.predicted_empty_line_counter[file_name] / self.predicted_lines[file_name], 4) * 100}%")

def predict_without_pretrain():
    from preprocessing.event_plan.event_extractor import EventExtractor

    process_list = []
    for dataset_name in ["roc-stories"]:
        event_predictor = EventPredictor(dataset_name=dataset_name,
                                         event_extractor_path=f"{BASE_DIR}/output/event-plan/cache/{dataset_name}_event_graph.pkl",
                                         cache_dir=f"{BASE_DIR}/output/event-plan/cache",
                                         data_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}",
                                         output_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}")
        target_num = 4 if dataset_name == "roc-stories" else 10
        # ps = Process(target=event_predictor.predict_for_file,
        #              args=("test.source.txt", "test_predicted_event.source.txt", target_num, target_num))
        # ps.start()
        # process_list.append(ps)
        # ps = Process(target=event_predictor.predict_for_file,
        #              args=("val.source.txt", "val_predicted_event.source.txt", target_num, target_num))
        # ps.start()
        # process_list.append(ps)

        # debug
        event_predictor.predict_for_file("test.source.txt", "test_predicted_event.source.txt", target_num, target_num)
        event_predictor.predict_for_file("val.source.txt", "val_predicted_event.source.txt", target_num, target_num)

    for ps in process_list:
        ps.join()

# def predict_with_pretrain():
#     process_list = []
#
#     for dataset_name in ["roc-stories", "writing-prompts"]:
#         event_predictor = EventPredictor(name=dataset_name,
#                                          event_extractor_path=f"{BASE_DIR}/output/event-plan/cache/{dataset_name}_event_graph.pkl",
#                                          cache_dir=f"{BASE_DIR}/output/event-plan/cache",
#                                          data_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}",
#                                          output_dir=f"{BASE_DIR}/resources/datasets/event-plan/{dataset_name}")
#         target_num = 4 if dataset_name == "roc-stories" else 10
#         ps = Process(target=event_predictor.predict_for_file,
#                      args=("test.source.txt", "test_pretrain_pred_event.source.txt", target_num, target_num))
#         ps.start()
#         process_list.append(ps)
#         ps = Process(target=event_predictor.predict_for_file,
#                      args=("val.source.txt", "val_pretrain_pred_event.source.txt", target_num, target_num))
#         ps.start()
#         process_list.append(ps)
#
#     for ps in process_list:
#         ps.join()

if __name__ == '__main__':
    from preprocessing.event_plan.event_extractor import EventExtractor

    predict_without_pretrain()
