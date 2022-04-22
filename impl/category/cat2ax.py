"""Cat2Ax approach as described in "Heist & Paulheim, 2019: Uncovering the Semantics of Wikipedia Categories"

The extraction is performed in three steps:
1) Identify candidate category sets that share a textual pattern
2) Find characteristic properties and types for candidate sets and combine them to patterns
3) Apply patterns to all categories to extract axioms
"""

from typing import Dict, Union, List, Tuple, Optional, Set
import utils
from tqdm import tqdm
from spacy.tokens import Doc
from utils import get_logger
from collections import defaultdict, Counter
import operator
import numpy as np
import multiprocessing as mp
import impl.dbpedia.heuristics as dbp_heur
import impl.category.category_set as cat_set
from impl.category.category_set import CandidateSet
import impl.util.nlp as nlp_util
from impl.util.rdf import RdfPredicate
from impl.dbpedia.ontology import DbpType, DbpObjectPredicate, DbpOntologyStore
from impl.dbpedia.resource import DbpResource, DbpEntity, DbpResourceStore
from impl.dbpedia.category import DbpCategory


PATTERN_CONF = utils.get_config('cat2ax.pattern_confidence')


class Axiom:
    def __init__(self, predicate, value, confidence: float):
        self.predicate = predicate
        self.value = value
        self.confidence = confidence

    def implies(self, other):
        return self.predicate == other.predicate and self.value == other.value

    def accepts_resource(self, res: DbpResource) -> bool:
        raise NotImplementedError("Please use the subclasses.")

    def rejects_resource(self, res: DbpResource) -> bool:
        raise NotImplementedError("Please use the subclasses.")


class TypeAxiom(Axiom):
    def __init__(self, value: DbpType, confidence: float):
        super().__init__(RdfPredicate.TYPE, value, confidence)

    def implies(self, other):
        return super().implies(other) or other.value in DbpOntologyStore.instance().get_transitive_supertypes(self.value, include_self=True)

    def accepts_resource(self, res: DbpResource) -> bool:
        return self.value in res.get_transitive_types()

    def rejects_resource(self, res: DbpResource) -> bool:
        return self.value in {dt for t in res.get_types() for dt in dbp_heur.get_direct_disjoint_types(t)}


class RelationAxiom(Axiom):
    def accepts_resource(self, res: DbpResource) -> bool:
        dbr = DbpResourceStore.instance()
        props = res.get_properties()
        return self.predicate in props and (self.value in props[self.predicate] or dbr.resolve_redirect(self.value) in props[self.predicate])

    def rejects_resource(self, res: DbpResource) -> bool:
        if not dbp_heur.is_functional_predicate(self.predicate):
            return False
        dbr = DbpResourceStore.instance()
        props = res.get_properties()
        return self.predicate in props and self.value not in props[self.predicate] and dbr.resolve_redirect(self.value) not in props[self.predicate]


def get_type_axioms(cat: DbpCategory) -> List[TypeAxiom]:
    """Return all type axioms created by the Cat2Ax approach."""
    return [a for a in get_axioms(cat) if isinstance(a, TypeAxiom)]


def get_axioms(cat: DbpCategory) -> List[Axiom]:
    """Return all axioms created by the Cat2Ax approach."""
    global __CATEGORY_AXIOMS__
    if '__CATEGORY_AXIOMS__' not in globals():
        __CATEGORY_AXIOMS__ = defaultdict(list, utils.load_cache('cat2ax_axioms'))
        if not __CATEGORY_AXIOMS__:
            raise ValueError('category/cat2ax: Axioms not initialised. Run axiom extraction before using them!')

    return __CATEGORY_AXIOMS__[cat]


def extract_category_axioms(valid_categories: Set[DbpCategory]):
    """Run extraction for the given graph with a confidence of `pattern_confidence`."""
    candidate_sets = cat_set.get_category_sets()
    patterns = _extract_patterns(valid_categories, candidate_sets)
    return _extract_axioms(valid_categories, patterns)


# --- PATTERN EXTRACTION ---

def _extract_patterns(valid_categories: Set[DbpCategory], candidate_sets: List[CandidateSet]) -> Dict[Tuple[tuple, tuple], dict]:
    """Return property/type patterns extracted from `category_graph` for each set in `candidate_sets`."""
    get_logger().debug('Extracting Cat2Ax patterns..')
    patterns = defaultdict(lambda: {'preds': defaultdict(list), 'types': defaultdict(list)})

    for parent, children, (first_words, last_words) in tqdm(candidate_sets, desc='category/cat2ax: Extracting patterns'):
        predicate_frequencies = defaultdict(list)
        type_frequencies = defaultdict(list)
        type_surface_scores = _get_type_surface_scores(first_words + last_words)

        categories_with_matches = {cat: _get_match_for_category(cat, first_words, last_words) for cat in children if cat in valid_categories}
        categories_with_matches = {cat: match for cat, match in categories_with_matches.items() if match}
        category_count = len(categories_with_matches)
        for cat, match in categories_with_matches.items():
            # compute predicate frequencies
            cat_stats = cat.get_statistics()
            possible_vals = _get_resource_surface_scores(match)
            for (pred, val), freq in cat_stats['property_frequencies'].items():
                if val in possible_vals:
                    predicate_frequencies[pred].append(freq * possible_vals[val])
            for t, tf in cat_stats['type_frequencies'].items():
                type_frequencies[t].append(tf * type_surface_scores[t])
        if predicate_frequencies:
            # pad frequencies to get the correct median
            predicate_frequencies = {pred: freqs + ([0] * (category_count-len(freqs))) for pred, freqs in predicate_frequencies.items()}
            pred, freqs = max(predicate_frequencies.items(), key=lambda x: np.median(x[1]))
            med = np.median(freqs)
            if med > 0:
                patterns[(tuple(first_words), tuple(last_words))]['preds'][pred].extend([med] * category_count)
        if type_frequencies:
            # pad frequencies to get the correct median
            type_frequencies = {t: freqs + ([0] * (category_count-len(freqs))) for t, freqs in type_frequencies.items()}
            max_median = max(np.median(freqs) for freqs in type_frequencies.values())
            types = {t for t, freqs in type_frequencies.items() if np.median(freqs) >= max_median}
            if max_median > 0:
                for t in types:
                    patterns[(tuple(first_words), tuple(last_words))]['types'][t].extend([max_median] * category_count)

    get_logger().debug(f'Extracted {len(patterns)} Cat2Ax patterns.')
    return patterns


def _get_match_for_category(cat: DbpCategory, first_words: tuple, last_words: tuple) -> str:
    """Return variable part of the category name."""
    doc = nlp_util.remove_by_phrase(cat.get_label())
    return doc[len(first_words):len(doc)-len(last_words)].text


def _get_resource_surface_scores(text: str) -> Dict[Union[str, DbpEntity], float]:
    """Return resource lexicalisation scores for the given text."""
    resource_surface_scores = {}
    if not text:
        return resource_surface_scores
    resource_surface_scores[text] = 1
    dbr = DbpResourceStore.instance()
    if dbr.has_resource_with_name(text):
        direct_match = dbr.resolve_redirect(dbr.get_resource_by_name(text))
        if isinstance(direct_match, DbpEntity):
            resource_surface_scores[direct_match] = 1
    for surface_match, frequency in sorted(dbr.get_surface_form_references(text).items(), key=operator.itemgetter(1)):
        resource_surface_scores[surface_match] = frequency
    return resource_surface_scores


def _get_type_surface_scores(words: list, lemmatize=True) -> Dict[DbpType, float]:
    """Return type lexicalisation scores for a given set of `words`."""
    lexicalisation_scores = Counter()
    dbo = DbpOntologyStore.instance()
    word_lemmas = [nlp_util.lemmatize_token(word_doc[0]) for word_doc in nlp_util.parse_texts(words)] if lemmatize else words
    for lemma in word_lemmas:
        for t, score in dbo.get_type_lexicalisations(lemma).items():
            if t == dbo.get_type_root():
                continue  # ignore obvious root type axiom
            lexicalisation_scores[t] += score
    type_surface_scores = defaultdict(float, {t: score / lexicalisation_scores.total() for t, score in lexicalisation_scores.items()})

    # make sure that exact matches get at least appropriate probability
    for lemma in word_lemmas:
        for word_type in dbo.get_types_for_label(lemma):
            min_word_type_score = 1 / len(words)
            type_surface_scores[word_type] = max(type_surface_scores[word_type], min_word_type_score)

    return type_surface_scores


# --- PATTERN APPLICATION ---


def _extract_axioms(valid_categories: Set[DbpCategory], patterns: Dict[tuple, dict]):
    """Return axioms extracted from `category_graph` by applying `patterns` to all categories."""
    get_logger().debug('Extracting Cat2Ax axioms..')
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

    cat_contexts = [(cat, front_pattern_dict, back_pattern_dict, enclosing_pattern_dict) for cat in valid_categories]
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        category_axioms = {cat: axioms for cat, axioms in tqdm(pool.imap_unordered(_extract_axioms_for_cat, cat_contexts, chunksize=10000), total=len(cat_contexts), desc='category/cat2ax: Extracting axioms')}
    category_axioms = {cat: axioms for cat, axioms in category_axioms.items() if axioms}  # filter out empty axioms

    get_logger().debug(f'Extracted {sum(len(axioms) for axioms in category_axioms.values())} axioms for {len(category_axioms)} categories.')
    return category_axioms


def _get_confidence_pattern_set(pattern_set: Dict[tuple, dict], has_front: bool, has_back: bool) -> Dict[tuple, dict]:
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
def _fill_dict(dictionary: dict, elements: list, leaf):
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


def _extract_axioms_for_cat(cat_context: tuple) -> Tuple[DbpCategory, List[Axiom]]:
    cat, front_pattern_dict, back_pattern_dict, enclosing_pattern_dict = cat_context
    cat_doc = nlp_util.remove_by_phrase(cat.get_label())
    cat_stats = cat.get_statistics()

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
        if isinstance(pred, DbpObjectPredicate):
            res_labels = {a[2]: a[2].get_label() for a in similar_prop_axioms}
            similar_prop_axioms = {a for a in similar_prop_axioms if all(res_labels[a[2]] == val or res_labels[a[2]] not in val for val in res_labels.values())}
        best_prop_axiom = max(similar_prop_axioms, key=operator.itemgetter(3))
        cat_axioms.append(RelationAxiom(best_prop_axiom[1], best_prop_axiom[2], best_prop_axiom[3]))

    best_type_axiom = None
    for type_axiom in sorted(cat_type_axioms, key=operator.itemgetter(3), reverse=True):
        if not best_type_axiom or type_axiom[2] in DbpOntologyStore.instance().get_transitive_subtypes(best_type_axiom[2]):
            best_type_axiom = type_axiom
    if best_type_axiom:
        cat_axioms.append(TypeAxiom(best_type_axiom[2], best_type_axiom[3]))

    return cat, cat_axioms


def _detect_and_apply_patterns(pattern_dict: dict, cat: DbpCategory, cat_doc: Doc, cat_stats: dict) -> Tuple[Optional[tuple], Optional[tuple]]:
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


def _detect_patterns(pattern_dict: dict, words: List[str]) -> Tuple[Optional[dict], Optional[Union[tuple, int]]]:
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


def _apply_patterns_to_cat(axiom_patterns: dict, cat: DbpCategory, cat_stats: dict, text_diff: str, words_same: list):
    """Return axioms by applying the best matching pattern to a category."""
    pred_patterns = axiom_patterns['preds']
    possible_values = _get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * pred_patterns[p] * possible_values[v] for (p, v), freq in cat_stats['property_frequencies'].items() if p in pred_patterns and v in possible_values}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    prop_axiom = None
    if max_prop_score >= PATTERN_CONF:
        pred, val = prop
        prop_axiom = (cat, pred, val, max_prop_score)

    type_patterns = axiom_patterns['types']
    type_surface_scores = _get_type_surface_scores(words_same)
    types_scores = {t: freq * type_patterns[t] * type_surface_scores[t] for t, freq in cat_stats['type_frequencies'].items() if t in type_patterns}
    t, max_type_score = max(types_scores.items(), key=operator.itemgetter(1), default=(None, 0))
    type_axiom = None
    if max_type_score >= PATTERN_CONF:
        type_axiom = (cat, RdfPredicate.TYPE, t, max_type_score)

    return prop_axiom, type_axiom
