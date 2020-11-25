"""Mapping of the category-list graph to DBpedia types (including resolution of disjointnesses that are created)."""

import impl.util.nlp as nlp_util
import impl.dbpedia.store as dbp_store
import impl.dbpedia.util as dbp_util
import impl.dbpedia.heuristics as dbp_heur
import impl.category.cat2ax as cat_axioms
from collections import defaultdict
import util


def find_conflicting_edges(graph, use_listpage_resources: bool) -> set:
    util.get_logger().debug('CaLiGraph: Removing conflicting edges in CaLiGraph..')
    conflicting_edges = set()
    head_subject_lemmas = graph.get_node_LHS()
    direct_mappings = {node: _find_dbpedia_parents(graph, node, use_listpage_resources, True) for node in graph.nodes}
    for node in graph.traverse_nodes_topdown():
        for child in graph.children(node):
            if head_subject_lemmas[node] == head_subject_lemmas[child]:
                continue
            parent_disjoint_types = {dt for t in direct_mappings[node] for dt in dbp_heur.get_disjoint_types(t)}
            child_types = set(direct_mappings[child])
            if child_types.intersection(parent_disjoint_types):
                conflicting_edges.add((node, child))
    util.get_logger().debug(f'CaLiGraph: Found {len(conflicting_edges)} to remove.')
    return conflicting_edges


def find_mappings(graph, use_listpage_resources: bool) -> dict:
    """Return mappings from nodes in `graph` to DBpedia types retrieved from axioms of the Cat2Ax approach."""
    util.get_logger().debug('CaLiGraph: Retrieving mappings from DBpedia to CaLiGraph..')
    mappings = {node: _find_dbpedia_parents(graph, node, use_listpage_resources, False) for node in graph.nodes}

    # apply complete transitivity to the graph in order to discover disjointnesses
    for node in graph.traverse_nodes_topdown():
        for parent in graph.parents(node):
            for t, score in mappings[parent].items():
                mappings[node][t] = max(mappings[node][t], score)

    # resolve basic disjointnesses
    for node in graph.traverse_nodes_topdown():
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
    for node in graph.traverse_nodes_bottomup():
        parent_types = {t for p in graph.parents(node) for t in mappings[p]}
        node_types = set(mappings[node]).difference(parent_types)
        mappings[node] = dbp_store.get_independent_types(node_types)

    return mappings


def resolve_disjointnesses(graph, use_listpage_resources: bool):
    """Resolve violations of disjointness axioms that are created through the mapping to DBpedia types."""
    util.get_logger().debug('CaLiGraph: Resolving disjointnesses in CaLiGraph from integrated DBpedia types..')
    for node in graph.traverse_nodes_topdown():
        parents = graph.parents(node)
        coherent_type_sets = _find_coherent_type_sets({t: 1 for t in graph.get_dbpedia_types(node, force_recompute=True)})
        if len(coherent_type_sets) > 1:
            transitive_types = {tt for ts in coherent_type_sets for t in ts for tt in dbp_store.get_transitive_supertype_closure(t)}
            direct_types = {t for t in _find_dbpedia_parents(graph, node, use_listpage_resources, False)}
            if not direct_types:
                # compute direct types by finding the best matching type from lex score
                lex_scores = _compute_type_lexicalisation_scores(graph, node)
                types = [{t: lex_scores[t] for t in ts} for ts in coherent_type_sets]
                types = [(ts, max(ts.values())) for ts in types]
                direct_types, score = max(types, key=lambda x: x[1])
                direct_types = set() if score == 0 else set(direct_types)
            # make sure that types induced by parts are integrated in direct types
            part_types = {t for t in graph.get_parts(node) if dbp_util.is_dbp_type(t)}
            direct_types = (direct_types | part_types).difference({dt for t in part_types for dt in dbp_heur.get_disjoint_types(t)})
            direct_types = {tt for t in direct_types for tt in dbp_store.get_transitive_supertype_closure(t)}

            invalid_types = transitive_types.difference(direct_types)
            new_parents = {p for p in parents if not invalid_types.intersection(graph.get_dbpedia_types(p))}
            graph._remove_edges({(p, node) for p in parents.difference(new_parents)})
            if not new_parents and direct_types:
                independent_types = dbp_store.get_independent_types(direct_types)
                new_parents = {n for t in independent_types for n in graph.get_nodes_for_part(t) if n != node}
                graph._add_edges({(p, node) for p in new_parents})


def _find_dbpedia_parents(graph, node: str, use_listpage_resources: bool, direct_resources_only: bool) -> dict:
    """Retrieve DBpedia types that can be used as parents for `node` based on axioms discovered for it.

    If `use_listpage_resources` is True, the resources that are extracted from list pages are also taken into account.
    """
    type_lexicalisation_scores = defaultdict(lambda: 0.1, _compute_type_lexicalisation_scores(graph, node))
    type_resource_scores = defaultdict(lambda: 0.0, _compute_type_resource_scores(graph, node, use_listpage_resources, direct_resources_only))

    overall_scores = {t: type_lexicalisation_scores[t] * type_resource_scores[t] for t in type_resource_scores if dbp_util.is_dbp_type(t)}
    max_score = max(overall_scores.values(), default=0)
    if max_score < util.get_config('cat2ax.pattern_confidence'):
        return defaultdict(float)

    mapped_types = {t: score for t, score in overall_scores.items() if score >= max_score}
    result = defaultdict(float)
    for t, score in mapped_types.items():
        for tt in dbp_store.get_transitive_supertype_closure(t):
            result[tt] = max(result[tt], score)

    result = defaultdict(float, {t: score for t, score in result.items() if not dbp_heur.get_disjoint_types(t).intersection(set(result))})
    return result


def _compute_type_lexicalisation_scores(graph, node: str) -> dict:
    head_subject_lemmas = nlp_util.get_head_subject_lemmas(graph.get_name(node))
    return cat_axioms._get_type_surface_scores(head_subject_lemmas, lemmatize=False)


def _compute_type_resource_scores(graph, node: str, use_listpage_resources: bool, direct_resources_only: bool) -> dict:
    node_resources = graph.get_direct_dbpedia_resources(node, use_listpage_resources)
    if not direct_resources_only or len([r for r in node_resources if dbp_store.get_types(r)]) < 5:
        node_resources.update({r for sn in graph.descendants(node) for r in graph.get_direct_dbpedia_resources(sn, use_listpage_resources)})
    node_resources = node_resources.intersection(dbp_store.get_resources())
    if len(node_resources) < 5:
        return {}  # better not return anything, if number of resources is too small
    type_counts = defaultdict(int)
    for res in node_resources:
        for t in dbp_store.get_transitive_types(res):
            type_counts[t] += 1
    return {t: count / len(node_resources) for t, count in type_counts.items()}


def _find_coherent_type_sets(dbp_types: dict) -> list:
    """Find biggest subset of types in `dbp_types` that does not violate any disjointness axioms."""
    coherent_sets = []
    disjoint_type_mapping = {t: set(dbp_types).intersection(dbp_heur.get_disjoint_types(t)) for t in dbp_types}
    for t, score in dbp_types.items():
        # add a type t to all coherent sets that do not contain disjoint types of t
        disjoint_types = disjoint_type_mapping[t]
        found_set = False
        for cs in coherent_sets:
            if not disjoint_types.intersection(set(cs)):
                cs[t] = score
                found_set = True
        # or create a new coherent set if t can not be added to any existing set
        if not found_set:
            coherent_sets.append({t: score})

    if len(coherent_sets) > 1:
        # check that a type is in all its valid sets (as types can be in multiple sets)
        for t, score in dbp_types.items():
            disjoint_types = disjoint_type_mapping[t]
            for cs in coherent_sets:
                if not disjoint_types.intersection(set(cs)):
                    cs[t] = score

        # remove types that exist in all sets, as they are not disjoint with anything
        # finally, add them as individual set in case that they have the highest score
        all_set_types = {}
        for t in dbp_types:
            if all(t in cs for cs in coherent_sets):
                for cs in coherent_sets:
                    all_set_types[t] = cs[t]
                    del cs[t]
        coherent_sets.append(all_set_types)
        coherent_sets = [cs for cs in coherent_sets if cs]  # remove possibly empty coherent sets

    return coherent_sets


def _remove_types_from_mapping(graph, mappings: dict, node: str, types_to_remove: set):
    """Remove `types_to_remove` from a mapping for a node in order to resolve disjointnesses."""
    types_to_remove = {tt for t in types_to_remove for tt in dbp_store.get_transitive_subtype_closure(t)}

    node_closure = {node} | graph.descendants(node)
    node_closure.update({a for n in node_closure for a in graph.ancestors(n)})
    for n in node_closure:
        mappings[n] = {t: score for t, score in mappings[n].items() if t not in types_to_remove}
