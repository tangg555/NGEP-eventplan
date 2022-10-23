"""
@Desc:
@Reference:
- named entities
https://spacy.io/usage/linguistic-features/#named-entities
- NER label scheme
CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT,
QUANTITY, TIME, WORK_OF_ART
@Notes:

"""


import spacy
from spacy import displacy
from spacy.tokens.doc import Doc


def get_named_entites(doc: Doc):
    """
    e.g.
    ent.text, ent.start_char, ent.end_char, ent.label_
    The Great Wall 0 14 FAC
    """
    result = []
    for ent in doc.ents:
        result.append((ent.text, ent.start, ent.end, ent.label_))
    return result


def show_entities(doc: Doc):
    print("==============================================")
    print(f"sent: {doc.text}")
    for ent in doc.ents:
        print(ent.text, ent.start, ent.end, ent.label_)


def server_display(doc: Doc):
    displacy.serve(doc, style="ent")

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")
    show_entities(nlp("I am not exited to take over the Great Wall."))
