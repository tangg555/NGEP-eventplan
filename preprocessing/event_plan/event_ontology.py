"""
@Desc:
@Reference:

@Notes:

"""

from typing import Dict, Set
import uuid
from uuid import uuid3
from collections import Counter

from src.utils.string_utils import rm_extra_spaces

class Event(object):
    def __init__(self, generation_models: str, event_info: dict = None):
        # verb
        self.generation_models = generation_models
        if event_info is None:
            self.event_info = {"trigger": [],
                               "modifiers": [],
                               "agents": [],
                               "comps": [],
                               }
        else:
            self.event_info = event_info
        self.degree = 0
        self.extracted_sents: Set[str] = set()

    def __repr__(self):
        return str(self.__dict__)

    @property
    def string(self):
        all_components = []
        for one in self.event_info.values():
            all_components += one
        all_components.sort(key=lambda x: x[1], reverse=False)  # sort according to indices
        return rm_extra_spaces(' '.join([one[0] for one in all_components]))

    @property
    def uuid(self) -> str:
        return str(uuid3(uuid.NAMESPACE_OID, self.string))

class EventGraph(object):
    event_s = "[EVENT_s]"
    event_sep = "[EVENT_sep]"
    event_e = "[EVENT_e]"

    def __init__(self, name: str):
        self.name = name
        self.events: Dict[Event.uuid, Event] = dict()
        self.prev_events: Dict[Event.uuid, Counter] = dict()
        self.next_events: Dict[Event.uuid, Counter] = dict()

    def string_to_uuid(self, event_string: str):
        return str(uuid3(uuid.NAMESPACE_OID, event_string))

    @property
    def nodes_num(self) -> int:
        return len(self.events)

    @property
    def triggers_num(self) -> int:
        triggers_ = []
        for event in self.events.values():
            triggers_.append(event.generation_models)
        triggers_ = Counter(triggers_)
        return len(triggers_)

    @property
    def edges_num(self) -> int:
        return sum([len(one)  for one in self.next_events.values()])

    @property
    def avg_degree(self) -> float:
        return round(self.edges_num / self.nodes_num, 2)

    def find_event(self, event_string: str):
        return self.events.get(self.string_to_uuid(event_string), None)

    def next_candidates(self, event_id: Event.uuid, limit=3):
        candidates = self.next_events[event_id].most_common(limit)
        string_counter = Counter([(self.events[id].string, freq) for id, freq in candidates])
        return string_counter

    def prev_candidates(self, event_id: Event.uuid, limit=3):
        candidates = self.prev_events[event_id].most_common(limit)
        string_counter = Counter([(self.events[id].string, freq) for id, freq in candidates])
        return string_counter

    def has_relation(self, subj_id: Event.uuid, obj_id: Event.uuid):
        if obj_id in self.next_events[subj_id] or obj_id in self.prev_events[subj_id]:
            return True
        return False
