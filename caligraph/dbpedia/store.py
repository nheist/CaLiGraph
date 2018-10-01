import util
import caligraph.util.rdf as rdf_util
from . import util as dbp_util
from collections import defaultdict, Counter
import networkx as nx
from typing import Optional
import functools


# DBpedia resources


def get_resources() -> set:
    return set(_get_resource_type_mapping())


def get_types(dbp_resource: str) -> set:
    return {t for t in _get_resource_type_mapping()[dbp_resource] if dbp_util.is_dbp_type(t)}


def get_transitive_types(dbp_resource: str) -> set:
    types = get_types(dbp_resource)
    transitive_types = types | {st for t in types for st in get_transitive_supertypes(t)}
    return {t for t in transitive_types if dbp_util.is_dbp_type(t)}


def get_properties(dbp_resource: str) -> set:
    return _get_resource_property_mapping()[dbp_resource]


def _get_resource_type_mapping() -> dict:
    global __RESOURCE_TYPE_MAPPING__
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [util.get_data_file('files.dbpedia.instance_types'), util.get_data_file('files.dbpedia.transitive_instance_types')]
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)

    return __RESOURCE_TYPE_MAPPING__


# DBpedia property


def get_property_frequency_distribution(dbp_property: str) -> dict:
    global __PROPERTY_FREQUENCY_DISTRIBUTION__
    if '__PROPERTY_FREQUENCY_DISTRIBUTION__' not in globals():
        __PROPERTY_FREQUENCY_DISTRIBUTION__ = util.load_or_create_cache('dbpedia_property_frequency_distribution', _compute_property_frequency_distribution)

    return __PROPERTY_FREQUENCY_DISTRIBUTION__[dbp_property]


def _compute_property_frequency_distribution() -> dict:
    property_frequency_distribution = defaultdict(functools.partial(defaultdict, int))
    for properties in _get_resource_property_mapping().values():
        for property, value in properties:
            property_frequency_distribution[property][value] += 1
    for prop, value_counts in property_frequency_distribution.items():
        property_frequency_distribution[prop]['_sum'] = sum(value_counts.values())
    return property_frequency_distribution


def _get_resource_property_mapping() -> dict:
    global __RESOURCE_PROPERTY_MAPPING__
    if '__RESOURCE_PROPERTY_MAPPING__' not in globals():
        property_files = [util.get_data_file('files.dbpedia.mappingbased_literals'), util.get_data_file('files.dbpedia.mappingbased_objects')]
        initializer = lambda: rdf_util.create_tuple_dict_from_rdf(property_files)
        __RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_resource_properties', initializer)

    return __RESOURCE_PROPERTY_MAPPING__


def _get_domain(dbp_property: str) -> Optional[str]:
    global __PROPERTY_DOMAIN__
    if '__PROPERTY_DOMAIN__' not in globals():
        __PROPERTY_DOMAIN__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DOMAIN)

    return __PROPERTY_DOMAIN__[dbp_property] if dbp_property in __PROPERTY_DOMAIN__ else None


# DBpedia types


def get_independent_types(dbp_types: set) -> set:
    """Returns only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_supertypes(dbp_type: str) -> set:
    type_graph = _get_type_graph()
    return set(type_graph.predecessors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_supertypes(dbp_type: str) -> set:
    global __TRANSITIVE_SUPERTYPE_MAPPING__
    if '__TRANSITIVE_SUPERTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUPERTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = nx.ancestors(type_graph, dbp_type) if dbp_type in type_graph else set()

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_subtypes(dbp_type: str) -> set:
    type_graph = _get_type_graph()
    return set(type_graph.successors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_subtypes(dbp_type: str) -> set:
    global __TRANSITIVE_SUBTYPE_MAPPING__
    if '__TRANSITIVE_SUBTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUBTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = nx.descendants(type_graph, dbp_type) if dbp_type in type_graph else set()

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_equivalent_types(dbp_type: str) -> set:
    global __EQUIVALENT_TYPE_MAPPING__
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS, reflexive=True)

    return {dbp_type} | __EQUIVALENT_TYPE_MAPPING__[dbp_type]


def get_disjoint_types(dbp_type: str) -> set:
    global __DISJOINT_TYPE_MAPPING__
    if '__DISJOINT_TYPE_MAPPING__' not in globals():
        __DISJOINT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DISJOINT_WITH, reflexive=True)
        # completing the subtype of each type with the subtypes of its disjoint types
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {t: {st for dt in disjoint_types for st in get_transitive_subtypes(dt)} for t, disjoint_types in __DISJOINT_TYPE_MAPPING__.items()})

    return __DISJOINT_TYPE_MAPPING__[dbp_type]


def get_type_depth(dbp_type: str) -> int:
    type_graph = _get_type_graph()
    global __TYPE_DEPTH__
    if '__TYPE_DEPTH__' not in globals():
        __TYPE_DEPTH__ = nx.shortest_path_length(type_graph, source=rdf_util.CLASS_OWL_THING)

    return __TYPE_DEPTH__[dbp_type] if dbp_type in __TYPE_DEPTH__ else 1


def get_type_frequency(dbp_type: str) -> float:
    global __TYPE_FREQUENCY__
    if '__TYPE_FREQUENCY__' not in globals():
        __TYPE_FREQUENCY__ = util.load_or_create_cache('dbpedia_resource_type_frequency', _compute_type_frequency)

    return __TYPE_FREQUENCY__[dbp_type] if dbp_type in __TYPE_FREQUENCY__ else 0


def _compute_type_frequency() -> dict:
    type_counts = sum([Counter(types) for types in _get_resource_type_mapping().values()], Counter())
    return {t: t_count / len(_get_resource_type_mapping()) for t, t_count in type_counts.items()}


def _get_type_graph() -> nx.DiGraph:
    global __TYPE_GRAPH__
    if '__TYPE_GRAPH__' not in globals():
        subtype_mapping = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # completing subtypes with subtypes of equivalent types
        subtype_mapping = {t: {est for et in get_equivalent_types(t) for st in subtype_mapping[et] for est in get_equivalent_types(st)} for t in set(subtype_mapping)}
        __TYPE_GRAPH__ = nx.DiGraph(incoming_graph_data=[(cat, subcat) for cat, subcats in subtype_mapping.items() for subcat in subcats])

    return __TYPE_GRAPH__
