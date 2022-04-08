from typing import Iterator, List, Tuple
from tqdm import tqdm
import utils
import spacy
from spacy.tokens import Span
from impl.util.spacy.components import LEXICAL_HEAD, LEXICAL_HEAD_SUBJECT, LEXICAL_HEAD_SUBJECT_PLURAL, BY_PHRASE
import impl.util.spacy.hearst_matcher as hearst_matcher


def _init_set_parser():
    set_parser = spacy.load('en_core_web_lg')
    set_parser.remove_pipe('ner')

    set_parser.vocab.strings.add(LEXICAL_HEAD)
    set_parser.vocab.strings.add(LEXICAL_HEAD_SUBJECT)
    set_parser.vocab.strings.add(LEXICAL_HEAD_SUBJECT_PLURAL)
    set_parser.vocab.strings.add(BY_PHRASE)

    set_parser.add_pipe('tag_lexical_head')
    set_parser.add_pipe('tag_lexical_head_subjects')
    set_parser.add_pipe('tag_by_phrase')
    return set_parser


BATCH_SIZE = 20000
N_PROCESSES = utils.get_config('max_cpus')

BASE_PARSER = spacy.load('en_core_web_lg')
SET_DOCUMENT_CACHE = {}
SET_PARSER = _init_set_parser()


def parse_sets(taxonomic_sets: list) -> Iterator:
    """Parse potential set structures of a taxonomy, e.g. Wikipedia categories or list pages."""
    unknown_sets = [s for s in taxonomic_sets if s and s not in SET_DOCUMENT_CACHE]
    if len(unknown_sets) <= BATCH_SIZE * N_PROCESSES:
        for s in unknown_sets:
            SET_DOCUMENT_CACHE[s] = SET_PARSER(s)
    else:
        set_tuples = [(s, s) for s in unknown_sets]
        for doc, s in tqdm(SET_PARSER.pipe(set_tuples, as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESSES), total=len(unknown_sets), desc='Parsing sets with spaCy'):
            SET_DOCUMENT_CACHE[s] = doc
    return iter([SET_DOCUMENT_CACHE[s] for s in taxonomic_sets])


def parse_texts(texts: list) -> Iterator:
    """Parse plain texts like the content of a Wikipedia page or a listing item."""
    if len(texts) <= BATCH_SIZE:
        return iter([BASE_PARSER(t) for t in texts])
    return iter(tqdm(BASE_PARSER.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROCESSES), total=len(texts), desc='Parsing texts with spaCy'))


def get_hearst_pairs(text: str) -> List[Tuple[Span, Span]]:
    """Parse text and retrieve (sub, obj) pairs for every occurrence of a hearst pattern."""
    return hearst_matcher.get_hearst_matches(text, BASE_PARSER)
