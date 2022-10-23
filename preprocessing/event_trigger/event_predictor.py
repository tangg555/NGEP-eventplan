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

import spacy
from tqdm import tqdm
import numpy as np

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from preprocessing.generation_models.event_extractor import EventExtractor
from preprocessing.generation_models.event_ontology import Event, EventGraph


class EventPredictor(EventExtractor):
    def __init__(self, name: str = "roc",
                 event_extractor_path=f"{BASE_DIR}/output/generation_models/cache/dataset_name_event_graph.pkl",
                 data_dir=f"{BASE_DIR}/resources/raw_data/dataset_name",
                 output_dir=f"{BASE_DIR}/resources/datasets/generation_models/dataset_name",
                 cache_dir=f"{BASE_DIR}/output/generation_models/cache",
                 nlp: spacy.language.Language = None):
        super().__init__(name, data_dir, output_dir, cache_dir, nlp)
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

    def frequency_based_inference(self, leading_event: Event, min_events: int, max_events: int, max_con_rept=2):
        event_list = []
        event_counter = Counter()
        lead_event_id = leading_event.uuid
        all_available_events = list(self.event_graph.events.keys())
        # continuously repeat events
        con_rept_count = 0

        if lead_event_id not in self.event_graph.next_events:
            # find event with same trigger
            event_candis_ = []
            for event_ in self.event_graph.events.values():
                if event_.generation_models == leading_event.generation_models:
                    event_candis_.append(event_)
            # update
            if len(event_candis_) > 0:
                leading_event_: Event = np.random.choice(event_candis_, size=1, replace=False)[0]
                lead_event_id = leading_event_.uuid
            else:
                # find a random one
                lead_event_id = np.random.choice(all_available_events, size=1, replace=False)[0]

        next_event_id = lead_event_id
        event_count = 0
        while True:
            candidate_counter_: Counter = self.event_graph.next_events[next_event_id].copy()
            # less than min_events: remove end
            if self.event_graph.event_e in candidate_counter_ and event_count < min_events:
                del candidate_counter_[self.event_graph.event_e]

            # select candidates for the next event
            candidates_ = list(candidate_counter_.keys())
            freq_ = list(candidate_counter_.values())  # frequency
            repeti_punishment_factors_ = []
            for candi_uuid in candidates_:
                factor = 1 if candi_uuid not in event_counter else \
                    float((1 / self.event_graph.events[candi_uuid].degree) *
                          (max_con_rept-event_counter[candi_uuid])/max_con_rept)
                repeti_punishment_factors_.append(factor)
            freq_ = freq_ * np.array(repeti_punishment_factors_, dtype=float)
            probs_ = np.array(freq_, dtype=float) / np.sum(freq_)
            if len(candidates_) > 0:
                next_event_id_temp = np.random.choice(candidates_, size=1, replace=False, p=probs_)[0]
            else:
                next_event_id_temp = np.random.choice(all_available_events, size=1, replace=False)[0]
            # count events continuously repeated
            if next_event_id == next_event_id_temp:
                con_rept_count += 1
            # state choice
            if next_event_id_temp == self.event_graph.event_e:
                break
            elif con_rept_count > max_con_rept:
                # if continuously repeated too much times then randomly choose another event
                next_event_id = np.random.choice(all_available_events, size=1, replace=False)[0]
                con_rept_count = 0
                continue
            else:
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
                line = self.rm_extra_spaces(line)
                line_doc = self.nlp(line)
                leading_event_list = []
                for sent_doc in line_doc.sents:
                    event = self.extract_event_from_sent(sent_doc)
                    leading_event_list.append(event)
                leading_event = leading_event_list[-1]
                event_list: list = self.frequency_based_inference(leading_event,
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
    process_list = []
    for dataset_name in ["roc-stories", "writing-prompts"]:
        event_predictor = EventPredictor(name=dataset_name,
                                         event_extractor_path=f"{BASE_DIR}/output/generation_models/cache/{dataset_name}_event_graph.pkl",
                                         cache_dir=f"{BASE_DIR}/output/generation_models/cache",
                                         data_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}",
                                         output_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}")
        target_num = 4 if dataset_name == "roc-stories" else 10
        ps = Process(target=event_predictor.predict_for_file,
                     args=("test.source.txt", "test_predicted_event.source.txt", target_num, target_num))
        ps.start()
        process_list.append(ps)
        ps = Process(target=event_predictor.predict_for_file,
                     args=("val.source.txt", "val_predicted_event.source.txt", target_num, target_num))
        ps.start()
        process_list.append(ps)

    for ps in process_list:
        ps.join()

# def predict_with_pretrain():
#     process_list = []
#
#     for dataset_name in ["roc-stories", "writing-prompts"]:
#         event_predictor = EventPredictor(name=dataset_name,
#                                          event_extractor_path=f"{BASE_DIR}/output/generation_models/cache/{dataset_name}_event_graph.pkl",
#                                          cache_dir=f"{BASE_DIR}/output/generation_models/cache",
#                                          data_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}",
#                                          output_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}")
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
    predict_without_pretrain()
