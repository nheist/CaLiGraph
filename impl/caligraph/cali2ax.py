"""Application of the Cat2Ax approach to CaLiGraph."""

from typing import Dict, Optional, Tuple, Any, List, Union, Set
from spacy.tokens import Doc
import utils
from utils import get_logger
from collections import defaultdict
import operator
import impl.category.category_set as cat_set
from impl.category.category_set import CandidateSet
from impl import category
import impl.util.nlp as nlp_util
import impl.category.cat2ax as cat_axioms
from impl.dbpedia.resource import DbpEntity
from impl.caligraph.ontology import ClgType, ClgPredicate, ClgObjectPredicate, ClgOntologyStore
from impl.caligraph.entity import ClgEntity, ClgEntityStore


def get_axiom_information() -> Dict[ClgType, Set[Tuple[ClgPredicate, Any, float]]]:
    axiom_information = utils.load_or_create_cache('cali2ax_axioms', _extract_axiom_information)
    return axiom_information


def _extract_axiom_information() -> Dict[ClgType, Set[Tuple[ClgPredicate, Any, float]]]:
    """Run extraction for the given graph reusing the category sets and patterns from the initial approach."""
    candidate_sets = cat_set.get_category_sets()
    patterns = _extract_patterns(candidate_sets)
    axioms = _extract_axioms(patterns)

    get_logger().info(f'Extracted {sum(len(axioms) for axioms in axioms.values())} axioms for {len(axioms)} categories.')
    return axioms


def _extract_patterns(candidate_sets: List[CandidateSet]) -> Dict[Tuple[tuple, tuple], dict]:
    clgo = ClgOntologyStore.instance()

    category_graph = category.get_conceptual_category_graph()
    dbp_patterns = cat_axioms._extract_patterns(category_graph, candidate_sets)

    patterns = {}
    for p, scores in dbp_patterns.items():
        preds = {clgo.get_predicate_for_dbp_predicate(p): meds for p, meds in scores['preds'].items()}
        patterns[p] = {'preds': preds}
    return patterns


def _extract_axioms(patterns: Dict[tuple, dict]) -> Dict[ClgType, Set[Tuple[ClgPredicate, Any, float]]]:
    """Run Cat2Ax axiom extraction on CaLiGraph."""
    clgo = ClgOntologyStore.instance()
    clge = ClgEntityStore.instance()

    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        cat_axioms._fill_dict(front_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        cat_axioms._fill_dict(back_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        cat_axioms._fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    axioms = defaultdict(set)
    for ct in clgo.get_types():
        property_frequencies = clge.get_property_frequencies(ct)
        labels = {r.get_label() for r in ct.get_associated_dbp_resources()}
        labels_without_by_phrases = [nlp_util.remove_by_phrase(label, return_doc=True) for label in labels]
        for label_doc in labels_without_by_phrases:
            node_axioms = []

            front_prop_axiom = _find_axioms(front_pattern_dict, ct, label_doc, property_frequencies)
            if front_prop_axiom:
                node_axioms.append(front_prop_axiom)

            back_prop_axiom = _find_axioms(back_pattern_dict, ct, label_doc, property_frequencies)
            if back_prop_axiom:
                node_axioms.append(back_prop_axiom)

            enclosing_prop_axiom = _find_axioms(enclosing_pattern_dict, ct, label_doc, property_frequencies)
            if enclosing_prop_axiom:
                node_axioms.append(enclosing_prop_axiom)

            prop_axioms_by_pred = {a[1]: {x for x in node_axioms if x[1] == a[1]} for a in node_axioms}
            for pred, similar_prop_axioms in prop_axioms_by_pred.items():
                if isinstance(pred, ClgObjectPredicate):
                    res_labels = {a[2]: a[2].get_label() for a in similar_prop_axioms}
                    similar_prop_axioms = {a for a in similar_prop_axioms if all(res_labels[a[2]] == val or res_labels[a[2]] not in val for val in res_labels.values())}
                best_prop_axiom = max(similar_prop_axioms, key=operator.itemgetter(3))
                axioms[ct].add(best_prop_axiom)

    return axioms


def _get_confidence_pattern_set(pattern_set: Dict[tuple, dict], has_front: bool, has_back: bool) -> Dict[tuple, dict]:
    """Return pattern confidences per pattern."""
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds_sum = sum(len(freqs) for freqs in axiom_patterns['preds'].values())
            result[pattern] = defaultdict(int, {p: len(freqs) / preds_sum for p, freqs in axiom_patterns['preds'].items()})
    return result


def _find_axioms(pattern_dict: dict, clg_type: ClgType, label_doc: Doc, property_frequencies: dict) -> Optional[tuple]:
    """Iterate over possible patterns to extract and return best axioms."""
    label_words = [w.text for w in label_doc]
    axiom_patterns, pattern_lengths = cat_axioms._detect_patterns(pattern_dict, label_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = label_doc[front_pattern_idx:back_pattern_idx].text
        return _get_axioms_for_node(axiom_patterns, clg_type, text_diff, property_frequencies)
    return None


def _get_axioms_for_node(axiom_patterns: dict, clg_type: ClgType, text_diff: str, property_frequencies: dict):
    """Return axioms by applying the best matching pattern to a node."""
    prop_axiom = None

    possible_values = _get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * axiom_patterns[p] * possible_values[v] for (p, v), freq in property_frequencies.items() if p in axiom_patterns and v in possible_values}
    (pred, val), max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= cat_axioms.PATTERN_CONF:
        prop_axiom = (clg_type, pred, val, max_prop_score)

    return prop_axiom


def _get_resource_surface_scores(text: str) -> Dict[Union[str, ClgEntity], float]:
    clge = ClgEntityStore.instance()

    dbp_resource_surface_scores = cat_axioms._get_resource_surface_scores(text)
    resource_surface_scores = {(clge.get_entity_for_dbp_entity(r) if isinstance(r, DbpEntity) else r): score for r, score in dbp_resource_surface_scores.items()}
    return resource_surface_scores
