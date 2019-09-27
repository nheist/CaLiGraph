import util
from collections import defaultdict
import operator
import numpy as np
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.category.category_set as cat_set
import impl.category.store as cat_store
import impl.util.nlp as nlp_util


PATTERN_CONF = util.get_config('cat2ax.pattern_confidence')


def extract_axioms(graph):
    candidate_sets = _extract_candidate_sets(graph)
    patterns = _extract_patterns(graph, candidate_sets)
    return _extract_axioms(graph, PATTERN_CONF, patterns)


# --- CANDIDATE SET EXTRACTION ---

def _extract_candidate_sets(graph) -> list:
    util.get_logger().debug('CaLi2Ax: Extracting candidate sets..')
    candidate_sets = []
    for node in graph.traverse_topdown():
        children = graph.children(node)
        children_docs = {c: nlp_util.parse(graph.get_label(c)) for c in children}
        # extract candidate sets ouf of direct children in caligraph
        candidate_sets.extend(cat_set._find_child_sets(node, children_docs))
        # extract candidate sets out of partitions of the children(from node parts that point to children)
        for part in graph.get_parts(node):
            part_children = cat_store.get_children(part)
            part_child_nodes = {n for pc in part_children for n in graph.get_nodes_for_part(pc) if n in children}
            if 2 < len(part_child_nodes) < len(children):
                part_child_docs = {n: children_docs[n] for n in part_child_nodes}
                candidate_sets.extend(cat_set._find_child_sets(node, part_child_docs))

    util.get_logger().debug(f'CaLi2Ax: Extracted {len(candidate_sets)} candidate sets.')
    return candidate_sets


# --- PATTERN EXTRACTION ---

def _extract_patterns(graph, candidate_sets: list) -> dict:
    util.get_logger().debug('CaLi2Ax: Extracting patterns..')
    patterns = defaultdict(lambda: defaultdict(list))

    for parent, children, (first_words, last_words) in candidate_sets:
        predicate_frequencies = defaultdict(list)

        children_with_matches = {c: _get_match_for_child(graph, c, first_words, last_words) for c in children}
        children_with_matches = {c: match for c, match in children_with_matches.items() if graph.has_node(c) and match}
        for child, match in children_with_matches.items():
            # compute predicate frequencies
            possible_vals = _get_resource_surface_scores(match)
            for (pred, val), freq in graph.get_property_frequencies(child).items():
                if val in possible_vals:
                    predicate_frequencies[pred].append(freq * possible_vals[val])
        if predicate_frequencies:
            # pad frequencies to get the correct median
            predicate_frequencies = {pred: freqs + ([0]*(len(children_with_matches)-len(freqs))) for pred, freqs in predicate_frequencies.items()}
            pred, freqs = max(predicate_frequencies.items(), key=lambda x: np.median(x[1]))
            med = np.median(freqs)
            if dbp_util.is_dbp_type(pred) and med > 0:
                patterns[(tuple(first_words), tuple(last_words))][pred].extend([med] * len(children_with_matches))

    util.get_logger().debug(f'CaLi2Ax: Extracted {len(patterns)} patterns.')
    return patterns


def _get_match_for_child(graph, child: str, first_words: tuple, last_words: tuple) -> str:
    doc = nlp_util.parse(graph.get_label(child))
    return doc[len(first_words):len(doc)-len(last_words)].text


def _get_resource_surface_scores(text):
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


# --- PATTERN APPLICATION ---

def _extract_axioms(graph, pattern_confidence, patterns):
    util.get_logger().debug('CaLi2Ax: Extracting axioms..')
    axioms = defaultdict(set)

    front_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, False).items():
        _fill_dict(front_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    back_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, False, True).items():
        _fill_dict(back_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    enclosing_pattern_dict = {}
    for (front_pattern, back_pattern), axiom_patterns in _get_confidence_pattern_set(patterns, True, True).items():
        _fill_dict(enclosing_pattern_dict, list(front_pattern), lambda d: _fill_dict(d, list(reversed(back_pattern)), axiom_patterns))

    for node in graph.nodes:
        node_doc = nlp_util.parse(graph.get_label(node))
        node_axioms = []

        front_prop_axiom = _find_axioms(graph, pattern_confidence, front_pattern_dict, node, node_doc)
        if front_prop_axiom:
            node_axioms.append(front_prop_axiom)

        back_prop_axiom = _find_axioms(graph, pattern_confidence, back_pattern_dict, node, node_doc)
        if back_prop_axiom:
            node_axioms.append(back_prop_axiom)

        enclosing_prop_axiom = _find_axioms(graph, pattern_confidence, enclosing_pattern_dict, node, node_doc)
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
    result = {}
    for pattern, axiom_patterns in pattern_set.items():
        if bool(pattern[0]) == has_front and bool(pattern[1]) == has_back:
            preds_sum = sum(len(freqs) for freqs in axiom_patterns.values())
            result[pattern] = defaultdict(lambda: 0, {p: len(freqs) / preds_sum for p, freqs in axiom_patterns.items()})
    return result


MARKER_HIT = '_marker_hit_'
MARKER_REVERSE = '_marker_reverse_'
def _fill_dict(dictionary, elements, leaf):
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


def _detect_pattern(pattern_dict, words):
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
            preds, back_pattern_length = _detect_pattern(ctx[MARKER_REVERSE], list(reversed(words)))
            return preds, (pattern_length, back_pattern_length)
        return None, None
    return None, None


def _get_axioms_for_node(graph, pattern_confidence, axiom_patterns, node, text_diff):
    prop_axiom = None

    possible_values = _get_resource_surface_scores(text_diff)
    props_scores = {(p, v): freq * axiom_patterns[p] * possible_values[v] for (p, v), freq in graph.get_property_frequencies(node).items() if p in axiom_patterns and v in possible_values}
    prop, max_prop_score = max(props_scores.items(), key=operator.itemgetter(1), default=((None, None), 0))
    if max_prop_score >= pattern_confidence:
        pred, val = prop
        prop_axiom = (node, pred, val, max_prop_score)

    return prop_axiom


def _find_axioms(graph, pattern_confidence, pattern_dict, node, node_doc):
    node_words = [w.text for w in node_doc]
    axiom_patterns, pattern_lengths = _detect_pattern(pattern_dict, node_words)
    if axiom_patterns:
        front_pattern_idx = pattern_lengths[0] or None
        back_pattern_idx = -1 * pattern_lengths[1] or None
        text_diff = node_doc[front_pattern_idx:back_pattern_idx].text
        return _get_axioms_for_node(graph, pattern_confidence, axiom_patterns, node, text_diff)
    return None
