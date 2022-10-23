"""
@Desc:
@Reference:
- Dependency label scheme
for em_core_web_lg: https://spacy.io/models/en#en_core_web_lg
ROOT, acl, acomp, advcl, advmod, agent, amod, appos, attr, aux, auxpass, case, cc, ccomp, compound,
conj, csubj, csubjpass, dative, dep, det, dobj, expl, intj, mark, meta, neg, nmod, npadvmod, nsubj,
 nsubjpass, nummod, oprd, parataxis, pcomp, pobj, poss, preconj, predet, prep, prt, punct, quantmod,
 relcl, xcomp

@Notes:
- verb(event trigger): root, prt, neg
root: the root of the sentence; prt: phrasal verb particle
- subj: csubj, csubjpass, nsubj, nsubjpass
csubj: clausal subject; nsubj: nominal subject
- comp: dobj, acomp, ccomp, xcomp, agent,
dobj: direct object; pobj: object of a preposition; iobj(dative): indirect object;
 acomp: adjectival complement; ccomp: clausal complement with internal subject;
xcomp: clausal complement with external subject
- agent: agent
- pickle load issue
https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
"""
import os
import sys
from multiprocessing import Process

import spacy
from typing import List
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from preprocessing.generation_models.dependency_parser import get_dependencies, show_dependencies
from preprocessing.generation_models.ner import get_named_entites
from preprocessing.generation_models.event_ontology import Event, EventGraph
from src.utils.file_utils import pickle_save, pickle_load


class EventExtractor(object):
    dep_columns = ["token", "index", "tag", "dep", "head", "head_index", "head_tag"]
    subject_tags = ["csubj", "csubjpass", "nsubj", "nsubjpass"]
    verb_phrase_tags = ["neg", "prt", 'ROOT']
    trigger_tags = ["ROOT"]
    modifier_tags = ["neg", "prt"]
    agent_tags = ["agent"]
    comp_tags = ["dobj", "acomp", "ccomp", "xcomp"]
    verb_pos = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    event_s = EventGraph.event_s
    event_sep = EventGraph.event_sep
    event_e = EventGraph.event_e

    def __init__(self, dataset_name: str = "roc",
                 data_dir=f"{BASE_DIR}/resources/raw_data/roc-stories",
                 output_dir=f"{BASE_DIR}/resources/datasets/generation_models/roc-stories",
                 cache_dir=f"{BASE_DIR}/output/generation_models/cache",
                 nlp: spacy.language.Language = None,
                 save_interval=0.1):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_file = self.data_dir.joinpath("train.source")
        self.nlp = spacy.load("en_core_web_lg") if nlp is None else nlp
        self.event_graph = EventGraph(self.dataset_name)
        self.progress = Counter()
        self.save_interval = save_interval
        self.save_path = self.cache_dir.joinpath(f"{self.dataset_name}_event_graph.pkl")

    def filter_with_trigger(self, filtered: pd.DataFrame, trigger: str, dep_df: pd.DataFrame):
        return pd.merge(dep_df[dep_df["head"].isin([trigger])], filtered, how='inner')

    def get_trigger(self, dep_df: pd.DataFrame):
        trig_df = dep_df[dep_df["dep"].isin(["ROOT"])]
        trigger = trig_df["token"].values[0]
        trigger_info = [(trigger, trig_df["index"].values[0])]
        return trigger, trigger_info

    def token_to_entity(self, subj_token, subj_index, entities):
        for ent, start, end, label in entities:
            if start <= subj_index and subj_index < end:
                return ent
        return subj_token

    def get_args(self, trigger: str, dep_df: pd.DataFrame, entities):
        modifiers = []
        agents = []
        comps = []
        mod_df = self.filter_with_trigger(dep_df[dep_df["dep"].isin(self.modifier_tags)], trigger, dep_df)
        for _, row in mod_df.iterrows():
            modifiers.append((row["token"], row["index"]))
        agent_df = self.filter_with_trigger(dep_df[dep_df["dep"].isin(self.agent_tags)], trigger, dep_df)
        for _, row in agent_df.iterrows():
            agents.append((row["token"], row["index"]))
        comp_df = self.filter_with_trigger(dep_df[dep_df["dep"].isin(self.comp_tags)], trigger, dep_df)
        for _, row in comp_df.iterrows():
            comps.append((row["token"], row["index"]))
        return modifiers, agents, comps

    def extract_event_from_sent(self, sent_doc):
        """
        dependencies: token.text, token.i, token.tag_, token.dep_, token.head.text, token.head.i, token.head.tag_
        entities: ent.text, ent.start_char, ent.end_char, ent.label_
        """
        dependencies = get_dependencies(sent_doc)
        entities = get_named_entites(sent_doc)
        dep_df = pd.DataFrame(dependencies, columns=self.dep_columns)
        trigger, trigger_info = self.get_trigger(dep_df)
        modifiers, agents, comps = self.get_args(trigger, dep_df, entities)
        event_info = {"trigger": trigger_info,
                      "modifiers": modifiers,
                      "agents": agents,
                      "comps": comps,
                      }
        event = Event(trigger, event_info)
        return event

    def save(self, extractor_path: str = None):
        if extractor_path is None:
            extractor_path = self.save_path
        else:
            self.save_path = extractor_path
        pickle_save(self, extractor_path)

    @classmethod
    def load(cls, extractor_path=None):
        return pickle_load(extractor_path)

    def rm_extra_spaces(self, input: str):
        return " ".join(input.split())

    def rm_sp_tokens(self, input: str):
        for sp in ["[", "]"]:
            input = input.replace(sp, "")
        return input

    def event_list_to_line(self, event_list: List[str]):
        if len(event_list) == 0:
            event_line = f"{self.event_s} {self.event_e}"
        else:
            event_line = f"{self.event_s} " + f" {self.event_sep} ".join(event_list) + f" {self.event_e}"
        return event_line

    def parse_file(self, file_name="corpus.txt"):
        file_path = self.data_dir.joinpath(file_name)
        with open(file_path, "r", encoding="utf-8") as fr:
            lines = [line.strip() for line in fr.readlines()]
            input_size = len(lines)
            existing_size = self.progress[file_path]
            rest_size = input_size - existing_size
            print(f"extracting file: {self.dataset_name} {file_name}, total: {input_size}, "
                  f"already finished: {existing_size}, rest: {input_size - existing_size}")
            for line in tqdm(lines[existing_size:], total=rest_size, desc=f"parsing the file inputs"):
                # remove [ and ] because of [MALE] [FEMALE] in roc stories, and some in writing prompts
                line = self.rm_sp_tokens(line)
                line = self.rm_extra_spaces(line)
                line_doc = self.nlp(line)
                pre_event = None
                for sent_doc in line_doc.sents:
                    event = self.extract_event_from_sent(sent_doc)
                    event_id = event.uuid
                    if event_id in self.event_graph.events:  # already has
                        event = self.event_graph.events[event_id]
                    else:
                        self.event_graph.events[event_id] = event
                        self.event_graph.prev_events[event_id] = Counter()
                        self.event_graph.next_events[event_id] = Counter()
                    event.degree += 1
                    event.extracted_sents.add(sent_doc.text)

                    # update relations
                    if pre_event is None:
                        self.event_graph.prev_events[event_id][EventGraph.event_s] += 1
                    else:
                        self.event_graph.prev_events[event_id][pre_event.uuid] += 1
                        self.event_graph.next_events[pre_event.uuid][event_id] += 1
                    pre_event = event
                self.event_graph.next_events[pre_event.uuid][EventGraph.event_e] += 1
                # end of line parsing
                existing_size += 1
                if existing_size % int(self.save_interval * input_size) == 0:
                    self.progress[file_path] = existing_size
                    print(f"parsing file progress: {self.progress[file_path]}, extractor saved")
                    self.save()
            # file finished
            self.progress[file_path] = existing_size
            print(f"parsing file progress: {self.progress[file_path]}, extractor saved")
            self.save()

    def merge(self, event_extractor):
        """
        EventGraph:
        self.events: Dict[UUID, Event] = dict()
        self.pre_events: Dict[UUID, Counter] = dict()
        self.next_events: Dict[UUID, Counter] = dict()
        """
        if not isinstance(event_extractor, EventExtractor):
            raise ValueError("input should be an object of EventExtractor")
        merged_graph = event_extractor.event_graph
        for event_id_, event_ in tqdm(merged_graph.events.items(),
                                      desc=f"merging events from {event_extractor.dataset_name}"):
            if event_id_ not in self.event_graph.events:
                self.event_graph.events[event_id_] = event_
        for event_id_, next_event_counter_ in tqdm(merged_graph.next_events.items(),
                                      desc=f"merging next_events from {event_extractor.dataset_name}"):
            if event_id_ not in self.event_graph.next_events:
                self.event_graph.next_events[event_id_] = Counter()
            self.event_graph.next_events[event_id_] += next_event_counter_
        for event_id_, prev_event_counter_ in tqdm(merged_graph.prev_events.items(),
                                      desc=f"merging prev_events from {event_extractor.dataset_name}"):
            if event_id_ not in self.event_graph.prev_events:
                self.event_graph.prev_events[event_id_] = Counter()
            self.event_graph.prev_events[event_id_] += prev_event_counter_
        return self

    @classmethod
    def merge_and_save(cls, output_path, extractor_dir, merged_files: list):
        """
        EventGraph:
        self.events: Dict[UUID, Event] = dict()
        self.pre_events: Dict[UUID, Counter] = dict()
        self.next_events: Dict[UUID, Counter] = dict()
        """
        new_extractor = None
        extractor_dir = Path(extractor_dir)
        if not extractor_dir.exists():
            raise ValueError(f"{extractor_dir} not exists")
        for one in merged_files:
            extractor: EventExtractor = cls.load(extractor_path=extractor_dir.joinpath(one))
            if new_extractor is None:
                new_extractor = extractor
                continue
            new_extractor.merge(extractor)
        new_extractor.save(extractor_path=output_path)
        return new_extractor

    @property
    def nodes_size(self):
        return len(self.event_graph.events)

    @property
    def edges_size(self):
        return sum([len(counter) for counter in self.event_graph.next_events.values()])

    @property
    def ave_edges_per_node(self):
        return round(self.edges_size/self.nodes_size, 2)

    @property
    def ave_node_degrees(self):
        return round(sum([event.degree for event in self.event_graph.events.values()]) / self.nodes_size, 2)

    @property
    def generation_modelss_size(self):
        triggers = set([event.generation_models for event in self.event_graph.events.values()])
        return len(triggers)

def ps_run(extractor, corpus_file):
    extractor.parse_file(file_name=corpus_file)

if __name__ == '__main__':
    process_list = []
    for dataset_name in ["roc-stories", "writing-prompts"]:
        save_path = f"{BASE_DIR}/output/generation_models/cache/{dataset_name}_event_graph.pkl"
        if os.path.exists(save_path):
            print(f"extractor loaded from {save_path}")
            event_extractor = EventExtractor.load(save_path)
        else:
            event_extractor = EventExtractor(dataset_name=dataset_name,
                                             cache_dir=f"{BASE_DIR}/output/generation_models/cache",
                                             data_dir=f"{BASE_DIR}/resources/datasets/generation_models/{dataset_name}")

        ps = Process(target=event_extractor.parse_file, args=("corpus.txt",))
        ps.start()
        process_list.append(ps)

    for ps in process_list:
        ps.join()