from collections import defaultdict
import util
import caligraph.util.rdf as rdf_util


def get_types(dbp_resource: str) -> set:
    return _get_resource_type_mapping()[dbp_resource]


def get_supertypes(dbp_type: str) -> set:
    return _get_supertype_mapping()[dbp_type]


def get_transitive_supertypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        direct_supertypes = get_supertypes(dbp_type)
        indirect_supertypes = {tst for dst in direct_supertypes for tst in get_transitive_supertypes(dst)}
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = direct_supertypes | indirect_supertypes

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_subtypes(dbp_type: str) -> set:
    return _get_subtype_mapping()[dbp_type]


def get_transitive_subtypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        direct_subtypes = get_subtypes(dbp_type)
        indirect_subtypes = {tst for dst in direct_subtypes for tst in get_transitive_subtypes(dst)}
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = direct_subtypes | indirect_subtypes

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_independent_types(dbp_types: set) -> set:
    """Returns only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_equivalent_types(dbp_type: str) -> set:
    return {dbp_type} | _get_equivalent_type_mapping()[dbp_type]


# BASIC GETTERS + INITIALIZERS
__TRANSITIVE_SUPERTYPE_MAPPING__ = defaultdict(set)
__TRANSITIVE_SUBTYPE_MAPPING__ = defaultdict(set)


def _get_resource_type_mapping() -> dict:
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [util.get_data_file('files.dbpedia.instance_types'), util.get_data_file('files.dbpedia.transitive_instance_types')]
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        global __RESOURCE_TYPE_MAPPING__
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)
    return __RESOURCE_TYPE_MAPPING__


def _get_supertype_mapping() -> dict:
    if '__SUPERTYPE_MAPPING__' not in globals():
        global __SUPERTYPE_MAPPING__
        __SUPERTYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF)
        # completing the supertypes of each type with the supertypes of its equivalent types
        for t in _get_equivalent_type_mapping():
            __SUPERTYPE_MAPPING__[t] = {st for et in get_equivalent_types(t) for st in __SUPERTYPE_MAPPING__[et]}
    return __SUPERTYPE_MAPPING__


def _get_subtype_mapping() -> dict:
    if '__SUBTYPE_MAPPING__' not in globals():
        global __SUBTYPE_MAPPING__
        __SUBTYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # completing the subtypes of each type with the subtypes of its equivalent types
        for t in _get_equivalent_type_mapping():
            __SUBTYPE_MAPPING__[t] = {st for et in get_equivalent_types(t) for st in __SUBTYPE_MAPPING__[et]}
    return __SUBTYPE_MAPPING__


def _get_equivalent_type_mapping() -> dict:
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        global __EQUIVALENT_TYPE_MAPPING__
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS)
    return __EQUIVALENT_TYPE_MAPPING__
