import util
import operator
import impl.util.rdf as rdf_util
from . import util as dbp_util
from collections import defaultdict
import networkx as nx
from typing import Optional, Tuple
import functools
import numpy as np


# DBpedia resources


def get_resources() -> set:
    return set(_get_label_mapping())


def get_label(dbp_object: str) -> str:
    labels = _get_label_mapping()
    return labels[dbp_object] if dbp_object in labels else dbp_util.object2name(dbp_object)


def _get_label_mapping() -> dict:
    global __RESOURCE_LABEL_MAPPING__
    if '__RESOURCE_LABEL_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.labels')], rdf_util.PREDICATE_LABEL)
        __RESOURCE_LABEL_MAPPING__ = util.load_or_create_cache('dbpedia_resource_labels', initializer)

    return __RESOURCE_LABEL_MAPPING__


def get_surface_score(surface_resource: str, dependent_resource: str) -> float:
    global __RESOURCE_SURFACE_SCORES__
    if '__RESOURCE_SURFACE_SCORES__' not in globals():
        __RESOURCE_SURFACE_SCORES__ = util.load_or_create_cache('dbpedia_resource_surface_scores', _compute_resource_surface_scores)

    if dbp_util.is_dbp_resource(surface_resource):
        for surface_form, score in sorted(__RESOURCE_SURFACE_SCORES__[surface_resource].items(), key=operator.itemgetter(1), reverse=True):
            if surface_form in dependent_resource.lower():
                return score
    elif dbp_util.is_dbp_type(surface_resource):
        if surface_resource[len(dbp_util.NAMESPACE_DBP_ONTOLOGY):].lower() in dependent_resource:
            return 1
    else:
        if surface_resource in dependent_resource:
            return 1
    return 0


def _compute_resource_surface_scores() -> dict:
    surface_scores = defaultdict(dict)
    for res in get_resources():
        redirect_res = resolve_redirect(res)
        surface_forms = {**get_surface_forms(res), **get_surface_forms(redirect_res)}
        total_mentions = sum(surface_forms.values())
        surface_scores[res] = {sf: mentions / total_mentions for sf, mentions in surface_forms.items()}
    return surface_scores


def get_surface_forms(dbp_resource: str) -> dict:
    global __RESOURCE_SURFACE_FORMS__
    if '__RESOURCE_SURFACE_FORMS__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_freq_dict_from_rdf([util.get_data_file('files.dbpedia.anchor_texts')], rdf_util.PREDICATE_ANCHOR_TEXT)
        __RESOURCE_SURFACE_FORMS__ = util.load_or_create_cache('dbpedia_resource_surface_forms', initializer)
    return __RESOURCE_SURFACE_FORMS__[dbp_resource]


def get_types(dbp_resource: str) -> set:
    return {t for t in _get_resource_type_mapping()[dbp_resource] if dbp_util.is_dbp_type(t)}


def _get_resource_type_mapping() -> dict:
    global __RESOURCE_TYPE_MAPPING__
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [
            util.get_data_file('files.dbpedia.instance_types'),
            util.get_data_file('files.dbpedia.transitive_instance_types'),
            util.get_data_file('files.dbpedia.instance_types_sdtyped'),
            util.get_data_file('files.dbpedia.instance_types_lhd'),
        ]
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)

    return __RESOURCE_TYPE_MAPPING__


def get_transitive_types(dbp_resource: str) -> set:
    transitive_types = {tt for t in get_types(dbp_resource) for tt in get_transitive_supertype_closure(t)}
    return {t for t in transitive_types if dbp_util.is_dbp_type(t)}


def get_properties(dbp_resource: str) -> dict:
    return get_resource_property_mapping()[dbp_resource]


def get_inverse_properties(dbp_resource: str) -> dict:
    return get_inverse_resource_property_mapping()[dbp_resource]


def get_interlanguage_links(dbp_resource: str) -> set:
    global __RESOURCE_INTERLANGUAGE_LINKS__
    if '__RESOURCE_INTERLANGUAGE_LINKS__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.interlanguage_links')], rdf_util.PREDICATE_SAME_AS, reflexive=True)
        __RESOURCE_INTERLANGUAGE_LINKS__ = util.load_or_create_cache('dbpedia_resource_interlanguage_links', initializer)

    return __RESOURCE_INTERLANGUAGE_LINKS__[dbp_resource]


def resolve_redirect(dbp_resource: str) -> str:
    global __REDIRECTS__
    if '__REDIRECTS__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.redirects')], rdf_util.PREDICATE_REDIRECTS)
        __REDIRECTS__ = util.load_or_create_cache('dbpedia_resource_redirects', initializer)

    return resolve_redirect(__REDIRECTS__[dbp_resource]) if dbp_resource in __REDIRECTS__ else dbp_resource


# DBpedia property


def get_property_frequency_distribution(dbp_predicate: str) -> dict:
    global __PROPERTY_FREQUENCY_DISTRIBUTION__
    if '__PROPERTY_FREQUENCY_DISTRIBUTION__' not in globals():
        __PROPERTY_FREQUENCY_DISTRIBUTION__ = util.load_or_create_cache('dbpedia_property_frequency_distribution', _compute_property_frequency_distribution)

    return __PROPERTY_FREQUENCY_DISTRIBUTION__[dbp_predicate]


def _compute_property_frequency_distribution() -> dict:
    property_frequency_distribution = defaultdict(functools.partial(defaultdict, int))
    for properties in get_resource_property_mapping().values():
        for pred, values in properties.items():
            for val in values:
                property_frequency_distribution[pred][val] += 1
    for pred, value_counts in property_frequency_distribution.items():
        property_frequency_distribution[pred]['_sum'] = sum(value_counts.values())
    return property_frequency_distribution


def get_resource_property_mapping() -> dict:
    global __RESOURCE_PROPERTY_MAPPING__
    if '__RESOURCE_PROPERTY_MAPPING__' not in globals():
        property_files = [util.get_data_file('files.dbpedia.mappingbased_literals'), util.get_data_file('files.dbpedia.mappingbased_objects')]
        initializer = lambda: rdf_util.create_dict_from_rdf(property_files)
        __RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_resource_properties', initializer)

    return __RESOURCE_PROPERTY_MAPPING__


def get_inverse_resource_property_mapping() -> dict:
    global __INVERSE_RESOURCE_PROPERTY_MAPPING__
    if '__INVERSE_RESOURCE_PROPERTY_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_dict_from_rdf([util.get_data_file('files.dbpedia.mappingbased_objects')], reverse_key=True)
        __INVERSE_RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_inverse_resource_properties', initializer)

    return __INVERSE_RESOURCE_PROPERTY_MAPPING__


def get_domain(dbp_predicate: str) -> Optional[str]:
    global __PREDICATE_DOMAIN__
    if '__PREDICATE_DOMAIN__' not in globals():
        __PREDICATE_DOMAIN__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DOMAIN)

    return __PREDICATE_DOMAIN__[dbp_predicate] if dbp_predicate in __PREDICATE_DOMAIN__ else None


def get_range(dbp_predicate: str) -> Optional[str]:
    global __PREDICATE_RANGE__
    if '__PREDICATE_RANGE__' not in globals():
        __PREDICATE_RANGE__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_RANGE)

    return __PREDICATE_RANGE__[dbp_predicate] if dbp_predicate in __PREDICATE_RANGE__ else None


def get_equivalent_predicates(dbp_predicate: str) -> set:
    global __EQUIVALENT_PREDICATE__
    if '__EQUIVALENT_PREDICATE__' not in globals():
        __EQUIVALENT_PREDICATE__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_PROPERTY)

    return __EQUIVALENT_PREDICATE__[dbp_predicate]


def is_functional(dbp_predicate: str) -> bool:
    global __PREDICATE_FUNCTIONAL__
    if '__PREDICATE_FUNCTIONAL__' not in globals():
        __PREDICATE_FUNCTIONAL__ = util.load_or_create_cache('dbpedia_functional_predicates', _create_functional_predicate_dict)

    return __PREDICATE_FUNCTIONAL__[dbp_predicate] if dbp_predicate in __PREDICATE_FUNCTIONAL__ else False


def _create_functional_predicate_dict():
    predicate_functionality = {pred: True for pred in get_all_predicates()}
    resource_property_mapping = get_resource_property_mapping()
    for r in resource_property_mapping:
        for pred in resource_property_mapping[r]:
            if len(resource_property_mapping[r][pred]) > 1:
                predicate_functionality[pred] = False

    return predicate_functionality


def get_all_predicates() -> set:
    global __PREDICATES__
    if '__PREDICATES__' not in globals():
        __PREDICATES__ = {pred for props in get_resource_property_mapping().values() for pred in props}

    return __PREDICATES__


# DBpedia types

def get_all_types() -> set:
    return set(_get_type_graph().nodes)


def get_cooccurrence_frequency(dbp_type: str, another_dbp_type: str) -> float:
    global __TYPE_COOCCURRENCE_FREQUENCY_MATRIX__, __TYPE_MATRIX_DICT__
    if '__TYPE_COOCCURRENCE_FREQUENCY_MATRIX__' not in globals():
        type_cooccurrence_matrix, __TYPE_MATRIX_DICT__ = util.load_or_create_cache('dbpedia_resource_type_cooccurrence_matrix', _create_resource_type_cooccurrence_matrix)
        max_occurrences = np.max(type_cooccurrence_matrix, axis=1).reshape(-1, 1)
        __TYPE_COOCCURRENCE_FREQUENCY_MATRIX__ = np.divide(type_cooccurrence_matrix, max_occurrences, out=np.zeros_like(type_cooccurrence_matrix, dtype=np.float32), where=(max_occurrences != 0))

    if dbp_type not in __TYPE_MATRIX_DICT__ or another_dbp_type not in __TYPE_MATRIX_DICT__:
        return 0

    return __TYPE_COOCCURRENCE_FREQUENCY_MATRIX__[__TYPE_MATRIX_DICT__[dbp_type], __TYPE_MATRIX_DICT__[another_dbp_type]]


def _create_resource_type_cooccurrence_matrix() -> Tuple[np.array, dict]:
    types = get_all_types()
    type_count = len(types)
    type_dict = {t: i for t, i in zip(types, range(type_count))}
    type_cooccurrence_matrix = np.zeros((type_count, type_count), dtype=np.int32)
    for r in get_resources():
        resource_type_indices = [type_dict[t] for t in get_transitive_types(r)]
        for i in resource_type_indices:
            type_cooccurrence_matrix[i, resource_type_indices] += 1
    return type_cooccurrence_matrix, type_dict


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


def get_transitive_supertype_closure(dbp_type: str) -> set:
    return {dbp_type} | get_transitive_supertypes(dbp_type)


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


def get_transitive_subtype_closure(dbp_type: str) -> set:
    return {dbp_type} | get_transitive_subtypes(dbp_type)


def get_equivalent_types(dbp_type: str) -> set:
    global __EQUIVALENT_TYPE_MAPPING__
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS, reflexive=True)

    return {dbp_type} | __EQUIVALENT_TYPE_MAPPING__[dbp_type]


def are_equivalent_types(dbp_types: set) -> bool:
    return dbp_types == get_equivalent_types(list(dbp_types)[0])


def get_disjoint_types(dbp_type: str) -> set:
    global __DISJOINT_TYPE_MAPPING__
    if '__DISJOINT_TYPE_MAPPING__' not in globals():
        __DISJOINT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DISJOINT_WITH, reflexive=True)
        # completing the subtype of each type with the subtypes of its disjoint types
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {t: {st for dt in disjoint_types for st in get_transitive_subtypes(dt)} for t, disjoint_types in __DISJOINT_TYPE_MAPPING__.items()})

    return __DISJOINT_TYPE_MAPPING__[dbp_type]


def get_type_depth(dbp_type: str) -> int:
    global __TYPE_DEPTH__
    if '__TYPE_DEPTH__' not in globals():
        type_graph = _get_type_graph()
        __TYPE_DEPTH__ = nx.shortest_path_length(type_graph, source=rdf_util.CLASS_OWL_THING)

    return __TYPE_DEPTH__[dbp_type] if dbp_type in __TYPE_DEPTH__ else 1


def get_type_frequency(dbp_type: str) -> float:
    global __TYPE_FREQUENCY__
    if '__TYPE_FREQUENCY__' not in globals():
        __TYPE_FREQUENCY__ = util.load_or_create_cache('dbpedia_resource_type_frequency', _compute_type_frequency)

    return __TYPE_FREQUENCY__[dbp_type] if dbp_type in __TYPE_FREQUENCY__ else 0


def _compute_type_frequency() -> dict:
    type_counts = rdf_util.create_count_dict(_get_resource_type_mapping().values())
    return {t: t_count / len(_get_resource_type_mapping()) for t, t_count in type_counts.items()}


def _get_type_graph() -> nx.DiGraph:
    global __TYPE_GRAPH__
    if '__TYPE_GRAPH__' not in globals():
        subtype_mapping = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # add missing types (i.e. those, that do not have subclasses at all)
        all_types = rdf_util.create_set_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS)
        subtype_mapping.update({et: set() for t in all_types for et in get_equivalent_types(t) if et not in subtype_mapping})
        # completing subtypes with subtypes of equivalent types
        subtype_mapping = {t: {est for et in get_equivalent_types(t) for st in subtype_mapping[et] for est in get_equivalent_types(st)} for t in set(subtype_mapping)}
        __TYPE_GRAPH__ = nx.DiGraph(incoming_graph_data=[(t, st) for t, sts in subtype_mapping.items() for st in sts])

    return __TYPE_GRAPH__
