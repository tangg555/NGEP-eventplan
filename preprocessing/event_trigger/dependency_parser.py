"""
@Desc:
@Reference:
- spacy models
https://spacy.io/models/en
e.g. python -m spacy download en_core_web_lg
- Dependency label scheme
for em_core_web_lg: https://spacy.io/models/en#en_core_web_lg
ROOT, acl, acomp, advcl, advmod, agent, amod, appos, attr, aux, auxpass, case, cc, ccomp, compound,
conj, csubj, csubjpass, dative, dep, det, dobj, expl, intj, mark, meta, neg, nmod, npadvmod, nsubj,
 nsubjpass, nummod, oprd, parataxis, pcomp, pobj, poss, preconj, predet, prep, prt, punct, quantmod,
 relcl, xcomp
- pos tag schema
for em_core_web_lg: https://spacy.io/models/en#en_core_web_lg
$, '', ,, -LRB-, -RRB-, ., :, ADD, AFX, CC, CD, DT, EX, FW, HYPH, IN, JJ, JJR, JJS, LS, MD, NFP,
NN, NNP, NNPS, NNS, PDT, POS, PRP, PRP$, RB, RBR, RBS, RP, SYM, TO, UH, VB, VBD, VBG, VBN, VBP,
VBZ, WDT, WP, WP$, WRB, XX, ``
- The meaning for those tags
DEP: https://universaldependencies.org/u/dep/
DEP: https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf
POS: https://cs.nyu.edu/~grishman/jet/guide/PennPOS.html
- Dependency Tag (pos)
https://www.ibm.com/docs/zh/wca/3.5.0?topic=analytics-part-speech-tag-sets
@Notes:
http://localhost:5000/ to find the visualized graph.
"""
import sys

import spacy
from spacy import displacy
from spacy.tokens.doc import Doc

def get_dependencies(doc: Doc):
    """
    e.g.
    token.text, token.tag_, token.dep_, token.head.text, token.head.tag_
    I(PRP) <-- nsubjpass -- exited(VBN)
    """
    result = []
    for token in doc:
        result.append((token.text, token.i, token.tag_, token.dep_, token.head.text, token.head.i, token.head.tag_))
    return result


def show_dependencies(doc: Doc):
    print("==============================================")
    print(f"sent: {doc.text}")
    for token in doc:
        print(f'{token.text}({token.i} {token.tag_}) <-- {token.dep_} -- '
              f'{token.head.text}({token.head.i} {token.head.tag_})')


def server_display(doc: Doc):
    displacy.serve(doc, style="dep")


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")
    # i could not afford to sleep in . i woke up suddenly to the bright sunlight .
    doc = nlp("she 'd refused to answer it then , and refused to answer it now .")
    show_dependencies(list(doc.sents)[0])
    server_display(list(doc.sents)[0])
