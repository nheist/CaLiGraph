from nltk.corpus import wordnet
from typing import Set
from collections import defaultdict
import util
import pickle
import impl.category.cat2ax as cat_axioms
import impl.category.nlp as cat_nlp
import impl.util.nlp as nlp_util


"""Compute hypernyms with similiar methods as Ponzetto et al.: Wikitaxonomy"""


THRESHOLD_AXIOM = 10
THRESHOLD_WIKI = 100
THRESHOLD_WEBISALOD = .4


def is_hypernym(hyper_word: str, hypo_word: str) -> bool:
    global __WIKITAXONOMY_HYPERNYMS__
    if '__WIKITAXONOMY_HYPERNYMS__' not in globals():
        __WIKITAXONOMY_HYPERNYMS__ = util.load_cache('wikitaxonomy_hypernyms')
        if not __WIKITAXONOMY_HYPERNYMS__:
            raise ValueError('wikitaxonomy_hypernyms not initialised. Run hypernym extraction once to create the necessary cache!')

    if is_synonym(hyper_word, hypo_word):
        return True
    return hyper_word in __WIKITAXONOMY_HYPERNYMS__[hypo_word]


def is_synonym(word: str, another_word: str) -> bool:
    return word == another_word or word in get_synonyms(another_word)


def get_synonyms(word: str) -> set:
    return {lm.name() for syn in wordnet.synsets(word) for lm in syn.lemmas()}


def compute_hypernyms(category_graph) -> dict:
    hypernyms = defaultdict(set)

    axiom_hypernyms = defaultdict(lambda: defaultdict(lambda: 0))
    for parent, child in _get_axiom_edges(category_graph):
        for cl in _get_headlemmas(child):
            for pl in _get_headlemmas(parent):
                axiom_hypernyms[cl][pl] += 1

    wiki_hypernyms = pickle.load(open('data_surface_forms/wiki_hypernyms_lemmas.p', mode='rb'))  # TODO: integrate

    webisalod_data = pickle.load(open('data_caligraph/webisalod_hypernyms.p', mode='rb'))  # TODO: integrate
    webisalod_hypernyms = defaultdict(dict)
    for parent, child, conf in webisalod_data:
        webisalod_hypernyms[child][parent] = conf

    candidates = set(axiom_hypernyms) | set(wiki_hypernyms) | set(webisalod_hypernyms)
    for candidate in candidates:
        hyper_count = defaultdict(lambda: 0)
        if candidate in axiom_hypernyms:
            for word, count in axiom_hypernyms[candidate].items():
                if count >= THRESHOLD_AXIOM:
                    hyper_count[word] += 2
        if candidate in wiki_hypernyms:
            for word, count in wiki_hypernyms[candidate].items():
                if count >= THRESHOLD_WIKI:
                    hyper_count[word] += 1
        if candidate in webisalod_hypernyms:
            for word, conf in webisalod_hypernyms[candidate].items():
                if conf >= THRESHOLD_WEBISALOD:
                    hyper_count[word] += 1
        hypernyms[candidate] = {word for word, count in hyper_count.items() if count > 1}

    return hypernyms


def _get_axiom_edges(category_graph) -> Set[tuple]:
    valid_axiom_edges = set()
    for parent in category_graph.nodes:
        parent_axioms = cat_axioms.get_axioms(parent)
        for child in category_graph.children(parent):
            child_axioms = cat_axioms.get_axioms(child)
            if not any(pa.contradicts(ca) for pa in parent_axioms for ca in child_axioms):
                if any(ca.implies(pa) for pa in parent_axioms for ca in child_axioms):
                    valid_axiom_edges.add((parent, child))
    return valid_axiom_edges


def _get_headlemmas(category: str) -> set:
    global __WIKITAXONOMY_CATEGORY_LEMMAS__
    if '__WIKITAXONOMY_CATEGORY_LEMMAS__' not in globals():
        __WIKITAXONOMY_CATEGORY_LEMMAS__ = {}

    if category not in __WIKITAXONOMY_CATEGORY_LEMMAS__:
        __WIKITAXONOMY_CATEGORY_LEMMAS__[category] = nlp_util.get_head_lemmas(cat_nlp.parse_category(category))

    return __WIKITAXONOMY_CATEGORY_LEMMAS__[category]
