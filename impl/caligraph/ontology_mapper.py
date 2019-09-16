from impl.caligraph.graph import CaLiGraph
import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.category.cat2ax as cat_axioms
from collections import defaultdict
import util


def find_mappings(graph: CaLiGraph, use_listpage_resources: bool) -> dict:
    mappings = {}
    for node in graph.children(graph.root_node):
        mappings.update(_find_mapping(graph, use_listpage_resources, node))
    return mappings


def _find_mapping(graph: CaLiGraph, use_listpage_resources: bool, node: str) -> dict:
    node_types = _find_dbpedia_parents(graph, use_listpage_resources, node)
    if node_types:
        return {node: node_types}
    else:
        mapping = {}
        for child in graph.children(node):
            mapping.update(_find_mapping(graph, use_listpage_resources, child))
        return mapping


def _find_dbpedia_parents(graph: CaLiGraph, use_listpage_resources: bool, node: str) -> set:
    name = graph.get_name(node)
    head_lemmas = nlp_util.get_head_lemmas(nlp_util.parse(name))
    type_lexicalisation_scores = defaultdict(lambda: 0.1, cat_axioms._get_type_surface_scores(head_lemmas))
    type_resource_scores = defaultdict(lambda: 0.0, _compute_type_resource_scores(graph, node, use_listpage_resources))

    overall_scores = {t: type_lexicalisation_scores[t] * type_resource_scores[t] for t in type_resource_scores if dbp_util.is_dbp_type(t)}
    max_score = max(overall_scores.values(), default=0)
    if max_score < util.get_config('cat2ax.pattern_confidence'):
        return set()

    mapped_types = {t for t, score in overall_scores.items() if score >= max_score}
    return dbp_store.get_independent_types(mapped_types)


def _compute_type_resource_scores(graph: CaLiGraph, node: str, use_listpage_resources: bool) -> dict:
    node_resources = {res for sn in ({node} | graph.descendants(node)) for res in graph.get_dbpedia_resources(sn, use_listpage_resources)}
    type_counts = defaultdict(int)
    for res in node_resources:
        for t in dbp_store.get_transitive_types(res):
            type_counts[t] += 1
    return {t: count / len(node_resources) for t, count in type_counts.items()}
