from collections import defaultdict
import util
import caligraph.util.rdf as rdf_util
import caligraph.util.dataframe as df_util
import pandas as pd


def get_types(dbp_resource: str) -> set:
    return df_util.get_active_columns(_get_resource_type_mapping(), {dbp_resource})


def get_supertypes(dbp_types: set) -> set:
    return df_util.get_active_columns(_get_supertype_mapping(), dbp_types)


def get_transitive_supertypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        direct_supertypes = get_supertypes({dbp_type})
        indirect_supertypes = {tst for dst in direct_supertypes for tst in get_transitive_supertypes(dst)}
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = direct_supertypes | indirect_supertypes

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_subtypes(dbp_types: set) -> set:
    return df_util.get_active_columns(_get_subtype_mapping(), dbp_types)


def get_transitive_subtypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        direct_subtypes = get_subtypes({dbp_type})
        indirect_subtypes = {tst for dst in direct_subtypes for tst in get_transitive_subtypes(dst)}
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = direct_subtypes | indirect_subtypes

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_independent_types(dbp_types: set) -> set:
    """Returns only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_equivalent_types(dbp_types: set) -> set:
    return dbp_types | df_util.get_active_columns(_get_equivalent_type_mapping(), dbp_types)


# BASIC GETTERS + INITIALIZERS
__TRANSITIVE_SUPERTYPE_MAPPING__ = defaultdict(set)
__TRANSITIVE_SUBTYPE_MAPPING__ = defaultdict(set)


def _get_resource_type_mapping() -> pd.DataFrame:
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [util.get_data_file('files.dbpedia.instance_types'), util.get_data_file('files.dbpedia.transitive_instance_types')]
        initializer = lambda: df_util.create_relation_frame_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        global __RESOURCE_TYPE_MAPPING__
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)
    return __RESOURCE_TYPE_MAPPING__


def _get_supertype_mapping() -> pd.DataFrame:
    if '__SUPERTYPE_MAPPING__' not in globals():
        global __SUPERTYPE_MAPPING__
        __SUPERTYPE_MAPPING__ = df_util.create_relation_frame_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF)
        # completing the supertypes of each type with the supertypes of its equivalent types
        __SUPERTYPE_MAPPING__.apply(lambda row: __SUPERTYPE_MAPPING__[__SUPERTYPE_MAPPING__.index.isin(get_equivalent_types({row.index}))].any(), axis=1, result_type='broadcast')
    return __SUPERTYPE_MAPPING__


def _get_subtype_mapping() -> pd.DataFrame:
    return _get_supertype_mapping().transpose()


def _get_equivalent_type_mapping() -> pd.DataFrame:
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        global __EQUIVALENT_TYPE_MAPPING__
        __EQUIVALENT_TYPE_MAPPING__ = df_util.create_relation_frame_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS)
    return __EQUIVALENT_TYPE_MAPPING__
