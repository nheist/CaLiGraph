"""Cat2Ax approach as described in "Heist & Paulheim, 2019: Uncovering the Semantics of Wikipedia Categories"

The extraction is performed in three steps:
1) Identify candidate category sets that share a textual pattern
2) Find characteristic properties and types for candidate sets and combine them to patterns
3) Apply patterns to all categories to extract axioms
"""

import utils
from utils import log_debug
from collections import defaultdict
from typing import List, Tuple
import operator
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.category_set as cat_set
import impl.category.store as cat_store
import impl.util.nlp as nlp_util
import impl.util.rdf as rdf_util


PATTERN_CONF = utils.get_config('cat2ax.pattern_confidence')


class Axiom:
    def __init__(self, predicate: str, value: str, confidence: float):
        self.predicate = predicate
        self.value = value
        self.confidence = confidence

    def implies(self, other):
        return self.predicate == other.predicate and self.value == other.value

    def contradicts(self, other):
        raise NotImplementedError("Please use the subclasses.")

    def accepts_resource(self, dbp_resource: str) -> bool:
        raise NotImplementedError("Please use the subclasses.")

    def rejects_resource(self, dbp_resource: str) -> bool:
        raise NotImplementedError("Please use the subclasses.")


class TypeAxiom(Axiom):
    def __init__(self, value: str, confidence: float):
        super().__init__(rdf_util.PREDICATE_TYPE, value, confidence)

    def implies(self, other):
        return super().implies(other) or other.value in dbp_store.get_transitive_supertype_closure(self.value)

    def accepts_resource(self, dbp_resource: str) -> bool:
        return self.value in dbp_store.get_transitive_types(dbp_resource)

    def rejects_resource(self, dbp_resource: str) -> bool:
        return self.value in {dt for t in dbp_store.get_types(dbp_resource) for dt in dbp_heur.get_direct_disjoint_types(t)}


class RelationAxiom(Axiom):
    def accepts_resource(self, dbp_resource: str) -> bool:
        props = dbp_store.get_properties(dbp_resource)
        return self.predicate in props and (self.value in props[self.predicate] or dbp_store.resolve_redirect(self.value) in props[self.predicate])

    def rejects_resource(self, dbp_resource: str) -> bool:
        if not dbp_store.is_functional(self.predicate):
            return False
        props = dbp_store.get_properties(dbp_resource)
        return self.predicate in props and self.value not in props[self.predicate] and dbp_store.resolve_redirect(self.value) not in props[self.predicate]


def get_type_axioms(cat: str) -> list:
    """Return all type axioms created by the Cat2Ax approach."""
    return [a for a in get_axioms(cat) if type(a) == TypeAxiom]


def get_axioms(cat: str) -> list:
    """Return all axioms created by the Cat2Ax approach."""
    global __CATEGORY_AXIOMS__
    if '__CATEGORY_AXIOMS__' not in globals():
        __CATEGORY_AXIOMS__ = defaultdict(list, utils.load_cache('cat2ax_axioms'))
        if not __CATEGORY_AXIOMS__:
            raise ValueError('CATEGORY/CAT2AX: Axioms not initialised. Run axiom extraction before using them!')

    return __CATEGORY_AXIOMS__[cat]


def extract_category_axioms(category_graph):
    """Run extraction for the given graph with a confidence of `pattern_confidence`."""
    candidate_sets = cat_set.get_category_sets()
    patterns = _extract_patterns(category_graph, candidate_sets)
    return _extract_axioms(category_graph, patterns)


# --- PATTERN EXTRACTION ---

def _extract_patterns(category_graph, candidate_sets):
    """Return property/type patterns extracted from `category_graph` for each set in `candidate_sets`."""
    log_debug('Extracting Cat2Ax patterns..')
    patterns = defaultdict(lambda: {'preds': defaultdict(list), 'types': defaultdict(list)})

    for parent, children, (first_words, last_words) in candidate_sets:
        predicate_frequencies = defaultdict(list)
        type_frequencies = defaultdict(list)
        type_surface_scores = _get_type_surface_scores(first_words + last_words)

        categories_with_matches = {cat: _get_match_for_category(cat, first_words, last_words) for cat in children}
        categories_with_matches = {cat: match for cat, match in categories_with_matches.items() if category_graph.has_node(cat) and match}
        for cat, match in categories_with_matches.items():
            # compute predicate frequencies
            statistics = cat_store.get_statistics(cat)
            possible_vals = _get_resource_surface_scores(match)
            for (pred, val), freq in statistics['property_frequencies'].items():
                if val in possible_vals:
                    predicate_frequencies[pred].append(freq * possible_vals[val])
            for t, tf in statistics['type_frequencies'].items():
                type_frequencies[t].append(tf * type_surface_scores[t])
        if predicate_frequencies:
            # pad frequencies to get the correct median
            predicate_frequencies = {pred: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for pred, freqs in predicate_frequencies.items()}
            pred, freqs = max(predicate_frequencies.items(), key=lambda x: np.median(x[1]))
            med = np.median(freqs)
            if dbp_util.is_dbp_type(pred) and med > 0:
                for _ in categories_with_matches:
                    patterns[(tuple(first_words), tuple(last_words))]['preds'][pred].append(med)
        if type_frequencies:
            # pad frequencies to get the correct median
            type_frequencies = {t: freqs + ([0]*(len(categories_with_matches)-len(freqs))) for t, freqs in type_frequencies.items()}
            max_median = max(np.median(freqs) for freqs in type_frequencies.values())
            types = {t for t, freqs in type_frequencies.items() if np.median(freqs) >= max_median}
            if max_median > 0:
                for _ in categories_with_matches:
                    for t in types:
                        patterns[(tuple(first_words), tuple(last_words))]['types'][t].append(max_median)

    log_debug(f'Extracted {len(patterns)} Cat2Ax patterns.')
    return patterns


def _get_match_for_category(category: str, first_words: tuple, last_words: tuple) -> str:
    """Return variable part of the category name."""
    doc = nlp_util.remove_by_phrase(cat_store.get_label(category))
    return doc[len(first_words):len(doc)-len(last_words)].text


def _get_resource_surface_scores(text):
    """Return resource lexicalisation scores for the given text."""
    resource_surface_scores = {}
    if not text:
        return resource_surface_scores
    resource_surface_scores[text] = 1
    direct_match = dbp_store.resolve_redirect(dbp_util.name2resource(text))
    if direct_match in dbp_store.get_resources():
        resource_surface_scores[direct_match] = 1
    for surface_match, frequency in sorted(dbp_store.get_inverse_lexicalisations(text.lower()).items(), key=operator.itemgetter(1)):
        resource_surface_scores[surface_match] = frequency
    return resource_surface_scores


def _get_type_surface_scores(words, lemmatize=True):
    """Return type lexicalisation scores for a given set of `words`."""
    lexicalisation_scores = defaultdict(int)
    word_lemmas = [nlp_util.lemmatize_token(word_doc[0]) for word_doc in nlp_util.parse_texts(words)] if lemmatize else words
    for lemma in word_lemmas:
        for t, score in dbp_store.get_type_lexicalisations(lemma).items():
            lexicalisation_scores[t] += score
    total_scores = sum(lexicalisation_scores.values())
    type_surface_scores = defaultdict(float, {t: score / total_scores for t, score in lexicalisation_scores.items()})

    # make sure that exact matches get at least appropriate probability
    for lemma in word_lemmas:
        word_types = dbp_store.get_types_by_name(lemma)
        for word_type in word_types:
            min_word_type_score = 1 / len(words)
            type_surface_scores[word_type] = max(type_surface_scores[word_type], min_word_type_score)

    return type_surface_scores


# --- PATTERN APPLICATION ---


def _extract_axioms(category_graph, patterns):
    """Return axioms extracted from `category_graph` by applying `patterns` to all categories."""
    log_debug('Extracting Cat2Ax axioms..')
    category_axioms = defaultdict(list)

    # process front/back/front+back patterns individually to reduce computational complexity
    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        _fill_dict(front_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        _fill_dict(back_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        _fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    cat_contexts = [(
        cat,
        nlp_util.remove_by_phrase(cat_store.get_label(cat)),
        cat_store.get_statistics(cat),
        front_pattern_dict, back_pattern_dict, enclosing_pattern_dict
    ) for cat in category_graph.content_nodes]

    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        category_axioms = {cat: axioms for cat, axioms in tqdm(pool.imap_unordered(_extract_axioms_for_cat, cat_contexts, chunksize=1000), total=len(cat_contexts), desc='CATEGORY/CAT2AX: Extracting axioms')}
    category_axioms = {cat: axioms for cat, axioms in category_axioms.items() if axioms}  # filter out empty axioms

    log_debug(f'Extracted {sum(len(axioms) for axioms in category_axioms.values())} axioms for {len(category_axioms)} categories.')
    return category_axioms


def _get_confidence_pattern_set(pattern_set, has_front, has_back):
    """Return pattern confidences per pattern."""
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds, types = axiom_patterns['preds'], axiom_patterns['types']
            preds_sum = sum(len(freqs) for freqs in preds.values())
            types_sum = sum(len(freqs) for freqs in types.values())
            result[pattern] = {
                'preds': defaultdict(int, {p: len(freqs) / preds_sum for p, freqs in preds.items()}),
                'types': defaultdict(int, {t: len(freqs) / types_sum for t, freqs in types.items()})
            }
    return result


MARKER_HIT = '_marker_hit_'
MARKER_REVERSE = '_marker_reverse_'
def _fill_dict(dictionary, elements, leaf):
    """Recursively fill a dictionary with a given sequence of elements and finally apply/append `leaf`."""
    if not elements:
        if callable(leaf):
            if MARKER_REVERSE not in dictionary:
                dictionary[MARKER_REVERSE] = {}
            leaf(dictionary[MARKER_REVERSE])
        else:
            dictionary[MARKER_HIT] = leaf
    else:
        if elements[0] not in dictionary:
            dictionary[elements[0]] = {}
        _fill_dict(dictionary[elements[0]], elements[1:], leaf)


def _extract_axioms_for_cat(cat_context: tuple) -> Tuple[str, List[Axiom]]:
    cat, cat_doc, cat_stats, front_pattern_dict, back_pattern_dict, enclosing_pattern_dict = cat_context

    # find all axiom candidates for cat
    cat_prop_axioms = []
    cat_type_axioms = []

    front_prop_axiom, front_type_axiom = _detect_and_apply_patterns(front_pattern_dict, cat, cat_doc, cat_stats)
    if front_prop_axiom:
        cat_prop_axioms.append(front_prop_axiom)
    if front_type_axiom:
        cat_type_axioms.append(front_type_axiom)

    back_prop_axiom, back_type_axiom = _detect_and_apply_patterns(back_pattern_dict, cat, cat_doc, cat_stats)
    if back_prop_axiom:
        cat_prop_axioms.append(back_prop_axiom)
    if back_type_axiom:
        cat_type_axioms.append(back_type_axiom)

    enclosing_prop_axiom, enclosing_type_axiom = _detect_and_apply_patterns(enclosing_pattern_dict, cat, cat_doc, cat_stats)
    if enclosing_prop_axiom:
        cat_prop_axioms.append(enclosing_prop_axiom)
    if enclosing_type_axiom:
        cat_type_axioms.append(enclosing_type_axiom)

    # select consistent set of axioms (multiple axioms can be selected if they do not contradict each other)
    cat_axioms = []
    prop_axioms_by_pred = {a[1]: {x for x in cat_prop_axioms if x[1] == a[1]} for a in cat_prop_axioms}
    for pred, similar_prop_axioms in prop_axioms_by_pred.items():
        if dbp_store.is_object_property(pred):
            res_labels = {a[2]: dbp_store.get_label(a[2]) for a in similar_prop_axioms}
            similar_prop_axioms = {a for a in similar_prop_axioms if all(res_labels[a[2]] == val or res_labels[a[2]] not in val for val in res_labels.values())}
        best_prop_axiom = max(similar_prop_axioms, key=operator.itemgetter(3))
        cat_axioms.append(RelationAxiom(best_prop_axiom[1], best_prop_axiom[2], best_prop_axiom[3]))

    best_type_axiom = None
    for type_axiom in sorted(cat_type_axioms, key=operator.itemgetter(3), reverse=True):
        if not best_type_axiom or type_axiom[2] in dbp_store.get_transitive_subtypes(best_type_axiom[2]):
            best_type_axiom = type_axiom
    if best_type_axiom:
        cat_axioms.append(TypeAxiom(best_type_axiom[2], best_type_axiom[3]))

    return cat, cat_axioms


def _detect_and_apply_patterns(pattern_dict, cat, cat_doc, cat_stats):
    """Iterate over possible patterns to extract and return best axioms."""
    cat_words = [w.text for w in cat_doc]
    axiom_patterns, pattern_lengths = _detect_patterns(pattern_dict, cat_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = cat_doc[front_pattern_idx:back_pattern_idx].text
        words_same = []
        if front_pattern_idx:
            words_same += cat_words[:front_pattern_idx]
        if back_pattern_idx:
            words_same += cat_words[back_pattern_idx:]
        return _apply_patterns_to_cat(axiom_patterns, cat, cat_stats, text_diff, words_same)
    return None, None


def _detect_patterns(pattern_dict, words):
    """Search for a pattern of `words` in `pattern_dict` and return if found - else return None."""
    pattern_length = 0
    ctx = pattern_dict
    for word in words:
        if word in ctx:
            ctx = ctx[word]
            pattern_length += 1
            continue
        if MARKER_HIT in ctx:
            return ctx[MARKER_HIT], pattern_length
        if MARKER_REVERSE in ctx:
            preds, back_pattern_length = _detect_patterns(ctx[MARKER_REVERSE], list(reversed(words)))
            return preds, (pattern_length, back_pattern_length)
        return None, None
    return None, None


def _apply_patterns_to_cat(axiom_patterns, cat, cat_stats, text_diff, words_same):
    """Return axioms by applying the best matching pattern to a category."""
    prop_axiom = None
    type_axiom = None

    pred_patterns = axiom_patterns['preds']
    possible_values = _get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * pred_patterns[p] * possible_values[v] for (p, v), freq in cat_stats['property_frequencies'].items() if p in pred_patterns and v in possible_values}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= PATTERN_CONF:
        pred, val = prop
        prop_axiom = (cat, pred, val, max_prop_score)

    type_patterns = axiom_patterns['types']
    type_surface_scores = _get_type_surface_scores(words_same)
    types_scores = {t: freq * type_patterns[t] * type_surface_scores[t] for t, freq in cat_stats['type_frequencies'].items() if t in type_patterns}
    t, max_type_score = max(types_scores.items(), key=operator.itemgetter(1), default=(None, 0))
    if max_type_score >= PATTERN_CONF:
        type_axiom = (cat, rdf_util.PREDICATE_TYPE, t, max_type_score)

    return prop_axiom, type_axiom
