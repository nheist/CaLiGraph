"""NLP methods mainly for the identification of head nouns of Wikipedia categories."""

import impl.util.spacy as spacy_util
import impl.util.string as str_util
from spacy.tokens import Doc, Token
from typing import Iterable, Iterator, Optional, Callable, Union
import re
import inflection


def without_stopwords(text: str) -> set:
    """Return the lemmatized versions of all non-stop-words in `text`."""
    text = remove_bracket_content(text.replace('-', ' '))
    return {lemmatize_token(word) for word in parse_set(text) if not word.is_stop}


def remove_bracket_content(text: str, bracket_type='(', substitute=' ') -> str:
    """Remove all parentheses from the given text."""
    if bracket_type == '(':
        pattern = r'\s*\([^()]*\)\s*'
    elif bracket_type == '[':
        pattern = r'\s*\[[^\[\]]*\]\s*'
    elif bracket_type == '<':
        pattern = r'\s*\<[^>]*\>+\s*'
    else:
        raise ValueError(f'Invalid bracket type "{bracket_type}" for the removal of bracket content.')
    return re.sub(pattern, substitute, text)


def get_canonical_name(text: str) -> str:
    """"Remove parts from the name that Wikipedia adds for organisational reasons (e.g. by-phrases or alphabetical splits)."""
    text = remove_by_phrase(text, return_doc=False)  # remove by-phrase
    text = re.sub(r'\s*/[A-Za-z]+:\s*[A-Za-z](\s*[-–]\s*[A-Za-z])?$', '', text)  # remove trailing alphabetical splits with slash and type, e.g. 'Fellows of the Royal Society/name: A-C'
    text = re.sub(r'\s+\([^()]+[-–][^()]+\)$', '', text)  # remove trailing parentheses with number or letter ranges, e.g. 'Interstate roads (1-10)'
    text = re.sub(r'\s+\([A-Z]\)$', '', text)  # remove trailing parentheses with single letter, e.g. 'Interstate roads (Y)'
    text = re.sub(r'\s*[-:,–]\s*[A-Z][a-z]*\s?[-–]\s?[A-Z][a-z]*$', '', text)  # remove trailing alphabetical ranges, e.g. 'Drugs: Sp-Sub'
    text = re.sub(r'\s*[-:–]\s*([A-Z],\s*)*[A-Z]$', '', text)  # remove trailing alphabetical splits, e.g. 'Football clubs in Sweden - Z' or '.. - X, Y, Z'
    text = re.sub(r'\s*/([A-Z],\s*)*[A-Z]$', '', text)  # remove trailing alphabetical splits with slash, e.g. 'Fellows of the Royal Society/A'
    text = re.sub(r'\s+([A-Z],\s*)+[A-Z]$', '', text)  # remove trailing alphabetical splits without indicator, e.g. 'Fellows of the Royal Society A, B, C'
    text = re.sub(r'\s*:\s*..?\s*[-–]\s*..?$', '', text)  # remove arbitrary trailing alphabetical splits, e.g. 'Fellows of the Royal Society: ! - K'
    return str_util.regularize_spaces(text).rstrip(',')


def get_lexhead_subjects(set_or_sets) -> Union[set, list]:
    """Return the lexical head subjects of `doc` as lemmas or lowercased text."""
    func = lambda doc: {lemmatize_token(w) for w in doc if w.ent_type_.startswith(spacy_util.LEXICAL_HEAD_SUBJECT)}
    return _process_one_or_many_sets(set_or_sets, func, default=set())


def has_plural_lexhead_subjects(set_or_sets) -> Union[bool, list]:
    """Return True, if the lexical head of `text` is in plural form."""
    singular_lexhead_func = lambda doc: any(w.ent_type_ == spacy_util.LEXICAL_HEAD_SUBJECT for w in doc)
    plural_lexhead_func = lambda doc: any(w.ent_type_ == spacy_util.LEXICAL_HEAD_SUBJECT_PLURAL for w in doc)
    plural_lexhead_only_func = lambda doc: plural_lexhead_func(doc) and not singular_lexhead_func(doc)
    return _process_one_or_many_sets(set_or_sets, plural_lexhead_only_func, default=False)


def get_lexhead_remainder(set_or_sets) -> Union[set, list]:
    """Return the non-subject part of the lexical head of `doc` as lemmas."""
    func = lambda doc: {w.lemma_ for w in doc if w.ent_type_ == spacy_util.LEXICAL_HEAD}
    return _process_one_or_many_sets(set_or_sets, func, default=set())


def get_nonlexhead_part(set_or_sets) -> Union[str, list]:
    """Return the words not contained in the lexical head of `doc`."""
    func = lambda doc: ''.join([w.text_with_ws for w in doc if not w.ent_type_.startswith(spacy_util.LEXICAL_HEAD)])
    return _process_one_or_many_sets(set_or_sets, func, default='')


def remove_by_phrase(text: str, return_doc=True):
    """Remove the 'by'-phrase at the end of a category or listpage, e.g. 'People by country' -> 'People'"""
    doc = parse_set(text)
    result = ''.join([w.text_with_ws for w in doc if w.ent_type_ != spacy_util.BY_PHRASE])
    return parse_set(result) if return_doc else result


def singularize_phrase(text: str) -> str:
    """Return the singular form of the phrase by looking for head nouns and converting them to the singular form."""
    doc = parse_set(text)
    result = doc.text.strip()
    if len(doc) == 1:
        return lemmatize_token(doc[0])
    for idx, w in enumerate(doc):
        if w.ent_type_ == spacy_util.LEXICAL_HEAD_SUBJECT_PLURAL:
            singularized_word = str_util.transfer_word_casing(w.text, lemmatize_token(w))
            result = result.replace(w.text, singularized_word)
            if len(doc) > idx+1 and doc[idx+1].text == 'and':
                result = result.replace('and', 'or')
    return result


def lemmatize_token(word: Token) -> str:
    if word.tag_ in ['NNS', 'NNPS'] and word.text == word.lemma_:
        return inflection.singularize(word.text)
    return word.lemma_


def _process_one_or_many_sets(set_or_sets, func: Callable, default=None):
    if set_or_sets is None:
        return default
    if type(set_or_sets) == str:
        return func(parse_set(set_or_sets))  # process single set
    return [func(doc) if doc else default for doc in parse_sets(set_or_sets)]  # process multiple sets


def parse_set(taxonomic_set: str) -> Optional[Doc]:
    return next(parse_sets([taxonomic_set]))


def parse_sets(taxonomic_sets: Iterable) -> Iterator[Optional[Doc]]:
    return spacy_util.parse_sets(list(taxonomic_sets))


def parse_text(text: str) -> Doc:
    return next(parse_texts([text]))


def parse_texts(texts: Iterable) -> Iterator[Doc]:
    return spacy_util.parse_texts(list(texts))
