from .graph import CategoryGraph
import caligraph.dbpedia.store as dbp_store
import caligraph.category.store as cat_store
from collections import defaultdict
from typing import Tuple


def get_metrics(graph: CategoryGraph) -> Tuple[float, float, float]:
    graph_resource_types = _get_graph_resource_type_mapping(graph)
    return _get_correctness(graph_resource_types), _get_accordance(graph_resource_types), _get_recall(graph_resource_types)


def _get_correctness(graph_resource_types: dict) -> float:
    """ 1 - Proportion of resources with disjoint types assigned """
    resources = dbp_store.get_resources()
    disjoint_type_resources = set()
    for r in resources:
        possible_disjoint_types = {dt for t in dbp_store.get_types(r) for dt in dbp_store.get_disjoint_types(t)}
        if possible_disjoint_types.intersection(graph_resource_types[r]):
            disjoint_type_resources.add(r)
    return 1 - (len(disjoint_type_resources) / len(resources))


def _get_accordance(graph_resource_types: dict) -> float:
    """ 1 - Proportion of resources with assigned types that are not subtypes of their currently most specific type """
    resources = dbp_store.get_resources()
    nonaccording_type_resources = set()
    for r in resources:
        independent_resource_types = dbp_store.get_independent_types(dbp_store.get_types(r))
        subtypes_of_most_specific_types = {tst for it in independent_resource_types for tst in dbp_store.get_transitive_subtypes(it)}
        possible_according_types = dbp_store.get_transitive_types(r) | subtypes_of_most_specific_types
        if graph_resource_types[r].difference(possible_according_types):
            nonaccording_type_resources.add(r)
    return 1 - (len(nonaccording_type_resources) / len(resources))


def _get_recall(graph_resource_types: dict) -> float:
    """ Proportion of correctly found types """
    found_types = 0
    overall_types = 0
    for r in dbp_store.get_resources():
        dbp_types = dbp_store.get_transitive_types(r)
        found_types += len(dbp_types.intersection(graph_resource_types[r]))
        overall_types += len(dbp_types)
    return found_types / overall_types


def _get_graph_resource_type_mapping(graph: CategoryGraph) -> dict:
    resources = dbp_store.get_resources()
    category_type_mapping = defaultdict(set, {cat: graph.dbp_types(cat) or set() for cat in graph.categories})
    resource_category_mapping = cat_store.get_resource_to_cats_mapping()
    return {res: {t for cat in resource_category_mapping[res] for t in category_type_mapping[cat]} for res in resources}
