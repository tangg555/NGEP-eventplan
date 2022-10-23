"""
@Desc:
@Reference:
@Notes:
"""

import sys
import spacy
from pathlib import Path
from multiprocessing import Process

from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from preprocessing.event_plan.event_extractor import EventExtractor


class EventAnnotator(EventExtractor):
    def __init__(self, name: str = "roc",
         data_dir=f"{BASE_DIR}/resources/raw_data/roc-stories",
         output_dir=f"{BASE_DIR}/resources/datasets/event-plan/roc-stories",
         cache_dir=f"{BASE_DIR}/output/event-plan/cache",
         nlp: spacy.language.Language = None):
        super().__init__(name, data_dir, output_dir, cache_dir, nlp)

    def annotate_file(self, input_file="train.target.txt", output_file="train_event.source.txt"):
        input_path = Path(self.data_dir).joinpath(input_file)
        output_path = Path(self.output_dir).joinpath(output_file)
        with open(input_path, "r", encoding="utf-8") as fr, \
            open(output_path, "a+", encoding="utf-8") as fa:
            input_lines = [line.strip() for line in fr.readlines()]
            input_size = len(input_lines)
            fa.seek(0)  # read from line 0
            existing_size = len(fa.readlines())
            rest_size = input_size - existing_size
            if input_size < existing_size:
                raise ValueError(f"input_size: {input_size} should < existing_size: {existing_size}")
            elif input_size == existing_size:
                return
            else:
                pass
            print(f"annotating file: {input_file}, total: {input_size}, "
                  f"already finished: {existing_size}, rest: {rest_size}")
            # annotating
            for line in tqdm(input_lines[existing_size:],
                                 total=rest_size,
                                 desc=f"annotating file {input_file}, and output to {output_file}"):
                # remove [ and ] because of [MALE] [FEMALE]
                line = self.rm_sp_tokens(line)
                line = self.rm_extra_spaces(line)
                line_doc = self.nlp(line)
                event_list = []
                for sent_doc in line_doc.sents:
                    event = self.extract_event_from_sent(sent_doc)
                    event_list.append(event.string)
                event_line = self.event_list_to_line(event_list)
                fa.write(event_line + "\n")

if __name__ == '__main__':
    process_list = []
    project_name = "event-plan"
    for dataset_name in ["roc-stories"]:
        event_annotator = EventAnnotator(name=dataset_name,
                                         cache_dir=f"{BASE_DIR}/output/{project_name}/cache",
                                         data_dir=f"{BASE_DIR}/resources/datasets/{project_name}/{dataset_name}",
                                         output_dir=f"{BASE_DIR}/resources/datasets/{project_name}/{dataset_name}")
        for prefix in ["train", "val", "test"]:
            ps = Process(target=event_annotator.annotate_file,
                         args=(f"{prefix}.target.txt", f"{prefix}_event.source.txt"))
            ps.start()
            process_list.append(ps)

    for ps in process_list:
        ps.join()

