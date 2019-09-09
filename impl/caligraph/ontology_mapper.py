from impl.caligraph.graph import CaLiGraph
import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.category.cat2ax as cat_axioms
from collections import defaultdict


def find_dbpedia_parent(graph: CaLiGraph, node: str, resource_weight=0) -> dict:
    name = graph.get_name(node)
    head_lemmas = nlp_util.get_head_lemmas(nlp_util.parse(name))
    type_lexicalisation_scores = cat_axioms._get_type_surface_scores(head_lemmas)
    type_resource_scores = defaultdict(lambda: resource_weight, _compute_type_resource_scores(graph, node))
    return {t: (type_lexicalisation_scores[t], type_resource_scores[t], type_lexicalisation_scores[t] * type_resource_scores[t]) for t in type_lexicalisation_scores}


def _compute_type_resource_scores(graph: CaLiGraph, node: str) -> dict:
    node_resources = {res for sn in ({node} | graph.descendants(node)) for res in graph.get_resources(sn)}
    type_counts = defaultdict(int)
    for res in node_resources:
        for t in dbp_store.get_transitive_types(res):
            type_counts[t] += 1
    return {t: count / len(node_resources) for t, count in type_counts.items()}
