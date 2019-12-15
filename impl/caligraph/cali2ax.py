"""Application of the Cat2Ax approach to the CaLiGraph."""

import util
from collections import defaultdict
import operator
import impl.dbpedia.store as dbp_store
import impl.category.category_set as cat_set
import impl.category.base as cat_base
import impl.util.nlp as nlp_util
import impl.category.cat2ax as cat_axioms
import impl.category.util as cat_util
import impl.list.util as list_util


def extract_axioms(graph) -> dict:
    """Run extraction for the given graph reusing the category sets and patterns from the initial approach."""
    category_graph = cat_base.get_conceptual_category_graph()
    candidate_sets = cat_set.get_category_sets()
    patterns = cat_axioms._extract_patterns(category_graph, candidate_sets)

    return _extract_axioms(graph, patterns)


def _extract_axioms(graph, patterns) -> dict:
    """Run actual axiom extraction on CaLiGraph."""
    util.get_logger().debug('CaLi2Ax: Extracting axioms..')
    axioms = defaultdict(set)

    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        cat_axioms._fill_dict(front_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        cat_axioms._fill_dict(back_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        cat_axioms._fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: cat_axioms._fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    for node in graph.nodes.difference({graph.root_node}):
        property_frequencies = graph.get_property_frequencies(node)
        for part in graph.get_parts(node):
            if cat_util.is_category(part):
                label = cat_util.category2name(part)
            elif list_util.is_listcategory(part) or list_util.is_listpage(part):
                label = list_util.list2name(part)
            else:
                continue

            node_doc = nlp_util.parse(label)
            node_axioms = []

            front_prop_axiom = _find_axioms(front_pattern_dict, node, node_doc, property_frequencies)
            if front_prop_axiom:
                node_axioms.append(front_prop_axiom)

            back_prop_axiom = _find_axioms(back_pattern_dict, node, node_doc, property_frequencies)
            if back_prop_axiom:
                node_axioms.append(back_prop_axiom)

            enclosing_prop_axiom = _find_axioms(enclosing_pattern_dict, node, node_doc, property_frequencies)
            if enclosing_prop_axiom:
                node_axioms.append(enclosing_prop_axiom)

            prop_axioms_by_pred = {a[1]: {x for x in node_axioms if x[1] == a[1]} for a in node_axioms}
            for pred, similar_prop_axioms in prop_axioms_by_pred.items():
                if dbp_store.is_object_property(pred):
                    res_labels = {a[2]: dbp_store.get_label(a[2]) for a in similar_prop_axioms}
                    similar_prop_axioms = {a for a in similar_prop_axioms if all(res_labels[a[2]] == val or res_labels[a[2]] not in val for val in res_labels.values())}
                best_prop_axiom = max(similar_prop_axioms, key=operator.itemgetter(3))
                axioms[node].add(best_prop_axiom)

    util.get_logger().debug(f'CaLi2Ax: Extracted {sum(len(axioms) for axioms in axioms.values())} axioms for {len(axioms)} categories.')
    return axioms


def _get_confidence_pattern_set(pattern_set, has_front, has_back):
    """Return pattern confidences per pattern."""
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds_sum = sum(len(freqs) for freqs in axiom_patterns['preds'].values())
            result[pattern] = defaultdict(lambda: 0, {p: len(freqs) / preds_sum for p, freqs in axiom_patterns['preds'].items()})
    return result


def _find_axioms(pattern_dict, node, node_doc, property_frequencies):
    """Iterate over possible patterns to extract and return best axioms."""
    node_words = [w.text for w in node_doc]
    axiom_patterns, pattern_lengths = cat_axioms._detect_pattern(pattern_dict, node_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = node_doc[front_pattern_idx:back_pattern_idx].text
        return _get_axioms_for_node(axiom_patterns, node, text_diff, property_frequencies)
    return None


def _get_axioms_for_node(axiom_patterns, node, text_diff, property_frequencies):
    """Return axioms by applying the best matching pattern to a node."""
    prop_axiom = None

    possible_values = cat_axioms._get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * axiom_patterns[p] * possible_values[v] for (p, v), freq in property_frequencies.items() if p in axiom_patterns and v in possible_values}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= cat_axioms.PATTERN_CONF:
        pred, val = prop
        prop_axiom = (node, pred, val, max_prop_score)

    return prop_axiom
