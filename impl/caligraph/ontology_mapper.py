import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
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

    for node in mappings:
        disjoint_types = {t: score for t, score in mappings[node].items() if set(mappings[node]).intersection(dbp_store.get_disjoint_types(t))}
        if disjoint_types:
            util.get_logger().debug('*******')
            util.get_logger().debug(f'Found disjoint types for node {node}')
            util.get_logger().debug('')
            for t, score in disjoint_types.items():
                util.get_logger().debug(f'{t}: {score}')

# TODO: resolve disjointnesses

    # remove transitivity from the mappings and create sets of types
    for parent, child in reversed(list(nx.bfs_edges(graph.graph, graph.root_node))):
        mappings[child] = set(mappings[child]).difference(set(mappings[parent]))

    return mappings


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
