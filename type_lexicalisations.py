"""Extraction of type lexicalisations from the Wikipedia corpus.
The resulting cache files are already placed in the cache folder but can be recomputed with this script.
"""

import util
from typing import Tuple
import re
import spacy
from spacy.matcher import Matcher
import pynif
import bz2
from collections import defaultdict
import impl.dbpedia.store as dbp_store


def extract_type_lexicalisations():
    """Crawl the Wikipedia corpus for hearst patterns that contain type lexicalisations and count the occurrences."""
    type_lexicalisations = defaultdict(lambda: defaultdict(int))
    wikipedia_hypernyms = defaultdict(lambda: defaultdict(int))

    nlp = spacy.load('en_core_web_lg')
    matcher = _init_pattern_matcher(nlp)
    for uri, plaintext in _retrieve_plaintexts():
        doc = nlp(plaintext)
        word_to_chunk_mapping = {word: chunk for chunk in doc.noun_chunks for word in chunk}
        for match in matcher(doc):
            # STEP 1: extract resource and lexicalisation from text
            match_id, start, end = match
            pattern_type = nlp.vocab.strings[match_id]
            res, lex = doc[start - 1], doc[end]
            res, lex = (lex, res) if patterns[pattern_type]['reverse'] else (res, lex)  # revert order based on pattern

            if res not in word_to_chunk_mapping or lex not in word_to_chunk_mapping:
                # discard, if resource/lexicalisation is not part of a proper noun chunk ( -> useless)
                continue
            res, lex = word_to_chunk_mapping[res], word_to_chunk_mapping[lex]

            # collect hypernym statistics in Wikipedia
            wikipedia_hypernyms[res.root.lemma_][lex.root.lemma_] += 1

            # STEP 2: for each word, count the types that it refers to
            if uri not in dbp_store.get_inverse_lexicalisations(res.text):
                # discard, if the resource text does not refer to the subject of the article
                continue

            for t in dbp_store.get_independent_types(dbp_store.get_types(uri)):
                for word in lex:
                    type_lexicalisations[word.lemma_][t] += 1

    wikipedia_hypernyms = {word: dict(hypernym_counts) for word, hypernym_counts in wikipedia_hypernyms.items()}
    util.update_cache('wikipedia_hypernyms', wikipedia_hypernyms)

    type_lexicalisations = {word: dict(type_counts) for word, type_counts in type_lexicalisations.items()}
    util.update_cache('dbpedia_type_lexicalisations', type_lexicalisations)


# WIKIPEDIA TEXT RETRIEVAL


def _retrieve_plaintexts() -> Tuple[str, str]:
    """Return an iterator over DBpedia resources and their Wikipedia plaintexts."""
    with bz2.open(util.get_data_file('files.dbpedia.nif_context'), mode='rb') as nif_file:
        nif_collection = pynif.NIFCollection.loads(nif_file.read(), format='turtle')
        for nif_context in nif_collection.contexts:
            resource_uri = nif_context.original_uri[:nif_context.original_uri.rfind('?')]
            # remove parentheses and line breaks from text for easier parsing
            resource_plaintext = _remove_parentheses_content(nif_context.mention.replace('\n', ' '))
            yield resource_uri, resource_plaintext


parentheses_matcher = re.compile(r' [\(\[].*?[\)\]]')
def _remove_parentheses_content(text: str) -> str:
    return parentheses_matcher.sub("", text)


# PATTERN MATCHING


patterns = {
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


def _init_pattern_matcher(nlp) -> Matcher:
    """Return a spacy matcher that is initialised with the hearst patterns defined in `patterns`."""
    matcher = Matcher(nlp.vocab)
    for k, vals in patterns.items():
        matcher.add(k, None, *vals['pattern'])
    return matcher


if __name__ == '__main__':
    extract_type_lexicalisations()
