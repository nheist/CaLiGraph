import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.cat2ax as cat_axioms
from collections import defaultdict
import util
import networkx as nx


def find_mappings(graph, use_listpage_resources: bool) -> dict:
    mappings = {node: _find_dbpedia_parents(graph, use_listpage_resources, node) for node in graph.nodes}

    # apply complete transitivity to the graph in order to discover disjointnesses
    for parent, child in nx.bfs_edges(graph.graph, graph.root_node):
        for t, score in mappings[parent].items():
            mappings[child][t] = max(mappings[child][t], score)

    # resolve disjointnesses
    for node, _ in nx.bfs_edges(graph.graph, graph.root_node):
        coherent_type_sets = _find_coherent_type_sets(mappings[node])
        if len(coherent_type_sets) <= 1:  # no disjoint sets
            continue

        coherent_type_sets = [(cs, max(cs.values())) for cs in coherent_type_sets]
        max_set_score = max([cs[1] for cs in coherent_type_sets])
        max_set_score_count = len([cs for cs in coherent_type_sets if cs[1] == max_set_score])
        if max_set_score_count > 1:  # no single superior set -> remove all type mappings
            types_to_remove = {t for cs in coherent_type_sets for t in cs[0]}
        else:  # there is one superior set -> remove types from all sets except for superior set
            types_to_remove = {t for cs in coherent_type_sets for t in cs[0] if cs[1] < max_set_score}
        _remove_types_from_mapping(graph, mappings, node, types_to_remove)

    # remove transitivity from the mappings and create sets of types
    for parent, child in reversed(list(nx.bfs_edges(graph.graph, graph.root_node))):
        mappings[child] = set(mappings[child]).difference(set(mappings[parent]))

    return mappings


def _find_coherent_type_sets(dbp_types: dict) -> list:
    coherent_sets = []
    disjoint_type_mapping = {t: set(dbp_types).intersection(dbp_heur.get_disjoint_types(t)) for t in dbp_types}
    for t, score in dbp_types.items():
        disjoint_types = disjoint_type_mapping[t]
        found_set = False
        for cs in coherent_sets:
            if not disjoint_types.intersection(set(cs)):
                cs[t] = score
        if not found_set:
            coherent_sets.extend({t: score})
    return coherent_sets


def _remove_types_from_mapping(graph, mappings: dict, node: str, types_to_remove: set):
    node_closure = {node} | graph.descendants(node)
    node_closure.update({a for n in node_closure for a in graph.ancestors(n)})
    for n in node_closure:
        mappings[n] = {t: score for t, score in mappings[n].items() if t not in types_to_remove}


def _find_dbpedia_parents(graph, use_listpage_resources: bool, node: str) -> dict:
    name = graph.get_name(node) or ''
    head_lemmas = nlp_util.get_head_lemmas(nlp_util.parse(name))
    type_lexicalisation_scores = defaultdict(lambda: 0.1, cat_axioms._get_type_surface_scores(head_lemmas))
    type_resource_scores = defaultdict(lambda: 0.0, _compute_type_resource_scores(graph, node, use_listpage_resources))

    overall_scores = {t: type_lexicalisation_scores[t] * type_resource_scores[t] for t in type_resource_scores if dbp_util.is_dbp_type(t)}
    max_score = max(overall_scores.values(), default=0)
    if max_score < util.get_config('cat2ax.pattern_confidence'):
        return defaultdict(float)

    mapped_types = {t: score for t, score in overall_scores.items() if score >= max_score}
    result = defaultdict(float)
    for t, score in mapped_types.items():
        for tt in dbp_store.get_transitive_supertype_closure(t):
            result[tt] = max(result[tt], score)
    return result


def _compute_type_resource_scores(graph, node: str, use_listpage_resources: bool) -> dict:
    node_resources = {res for sn in ({node} | graph.descendants(node)) for res in graph.get_dbpedia_resources(sn, use_listpage_resources)}
    type_counts = defaultdict(int)
    for res in node_resources:
        for t in dbp_store.get_transitive_types(res):
            type_counts[t] += 1
    return {t: count / len(node_resources) for t, count in type_counts.items()}
