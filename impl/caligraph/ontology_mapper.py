from impl.caligraph.graph import CaLiGraph
import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.category.cat2ax as cat_axioms
from collections import defaultdict


def find_dbpedia_parent(graph: CaLiGraph, node: str) -> dict:
    name = graph.get_name(node)
    head_lemmas = nlp_util.get_head_lemmas(nlp_util.parse(name))
    type_lexicalisation_scores = defaultdict(int, cat_axioms._get_type_surface_scores(head_lemmas))
    type_resource_scores = defaultdict(int, _compute_type_resource_scores(graph, node))

    if not type_lexicalisation_scores:
        return {t: (0, score, score) for t, score in type_resource_scores.items()}
    if not type_resource_scores:
        return {t: (score, 0, score) for t, score in type_lexicalisation_scores.items()}

    return {t: (type_lexicalisation_scores[t], type_resource_scores[t], type_lexicalisation_scores[t] * type_resource_scores[t]) for t in (set(type_lexicalisation_scores) | set(type_resource_scores))}


def _compute_type_resource_scores(graph: CaLiGraph, node: str) -> dict:
    node_resources = {res for sn in ({node} | graph.descendants(node)) for res in graph.get_resources(sn)}
    type_counts = defaultdict(int)
    for res in node_resources:
        for t in dbp_store.get_transitive_types(res):
            type_counts[t] += 1
    return {t: count / len(node_resources) for t, count in type_counts.items()}
