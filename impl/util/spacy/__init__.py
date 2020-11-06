import spacy
from impl.util.spacy.components import tag_lexical_head, tag_lexical_head_subjects, tag_by_phrase
import util
from typing import Iterator


BATCH_SIZE = 100
N_PROCESSES = util.get_config('max_cpus')


__SET_DOCUMENT_CACHE__ = {}


def parse_sets(taxonomic_sets: list) -> Iterator:
    """Parse potential set structures of a taxonomy, e.g. Wikipedia categories or list pages."""
    global __SET_PARSER__
    if '__SET_PARSER__' not in globals():
        __SET_PARSER__ = spacy.load('en_core_web_lg')
        __SET_PARSER__.remove_pipe('ner')
        __SET_PARSER__.add_pipe(tag_lexical_head, name='lexhead')
        __SET_PARSER__.add_pipe(tag_lexical_head_subjects, name='lexheadsub')
        __SET_PARSER__.add_pipe(tag_by_phrase, name='byphrase')
    if len(taxonomic_sets) <= BATCH_SIZE:
        for s in taxonomic_sets:
            if s not in __SET_DOCUMENT_CACHE__:
                __SET_DOCUMENT_CACHE__[s] = __SET_PARSER__(s)
        return iter([__SET_DOCUMENT_CACHE__[s] for s in taxonomic_sets])
    return __SET_PARSER__.pipe(taxonomic_sets, batch_size=BATCH_SIZE, n_process=N_PROCESSES)


def parse_texts(texts: list) -> Iterator:
    """Parse plain texts like the content of a Wikipedia page or a listing item."""
    global __TEXT_PARSER__
    if '__TEXT_PARSER__' not in globals():
        __TEXT_PARSER__ = spacy.load('en_core_web_lg')
    if len(texts) <= BATCH_SIZE:
        return iter([__TEXT_PARSER__(t) for t in texts])
    return __TEXT_PARSER__.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROCESSES)
