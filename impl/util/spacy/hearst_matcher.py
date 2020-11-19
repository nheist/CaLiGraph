from spacy.matcher import Matcher
from spacy.tokens import Span
from typing import List, Tuple


HEARST_PATTERNS = {
    'is-a': {
        'pattern': [[{'LEMMA': 'be'}, {'LOWER': 'a'}], [{'LEMMA': 'be'}, {'LOWER': 'an'}]],
        'reverse': False
    },
    'and-or-other': {
        'pattern': [[{'LOWER': 'and'}, {'LOWER': 'other'}], [{'LOWER': 'or'}, {'LOWER': 'other'}]],
        'reverse': False
    },
    'including': {
        'pattern': [[{'LOWER': 'including'}]],
        'reverse': True
    },
    'such-as': {
        'pattern': [[{'LOWER': 'such'}, {'LOWER': 'as'}]],
        'reverse': True
    },
    'especially': {
        'pattern': [[{'LOWER': 'especially'}]],
        'reverse': True
    },
    'particularly': {
        'pattern': [[{'LOWER': 'particularly'}]],
        'reverse': True
    },
    'other-than': {
        'pattern': [[{'LOWER': 'other'}, {'LOWER': 'than'}]],
        'reverse': True
    }
}


def get_hearst_matches(text: str, nlp) -> List[Tuple[Span, Span]]:
    doc = nlp(text)
    word_to_chunk_mapping = {word: chunk for chunk in doc.noun_chunks for word in chunk}
    matches = []

    matcher = _get_hearst_matcher(nlp)
    for match_id, start, end in matcher(doc):
        if len(doc) <= end:
            continue  # pattern occurs at the end of the sentence -> there is no object of the hearst pattern

        pattern_type = nlp.vocab.strings[match_id]
        sub, obj = doc[start - 1], doc[end]
        sub, obj = (obj, sub) if HEARST_PATTERNS[pattern_type]['reverse'] else (sub, obj)

        if sub not in word_to_chunk_mapping or obj not in word_to_chunk_mapping:
            # discard, if subject or object are no part of a proper noun chunk (-> useless discovery)
            continue
        matches.append((word_to_chunk_mapping[sub], word_to_chunk_mapping[obj]))
    return matches


def _get_hearst_matcher(nlp) -> Matcher:
    global __HEARST_MATCHER__
    if '__HEARST_MATCHER__' not in globals():
        __HEARST_MATCHER__ = Matcher(nlp.vocab)
        for k, vals in HEARST_PATTERNS.items():
            __HEARST_MATCHER__.add(k, None, *vals['pattern'])
    return __HEARST_MATCHER__
