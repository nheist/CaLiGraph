from collections import defaultdict
import util
import caligraph.util.rdf as rdfutil
import pandas as pd


def get_types_for_resource(dbp_resource: str) -> set:
    return set(_get_resource_type_mapping()[dbp_resource])


def get_supertypes(dbp_type: str) -> set:
    supertypes = _get_supertype_mapping()[dbp_type]
    return supertypes | {et for t in supertypes for et in _get_equivalent_type_mapping()[t]}


def get_transitive_supertypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        direct_supertypes = get_supertypes(dbp_type)
        indirect_supertypes = {tst for dst in direct_supertypes for tst in get_transitive_supertypes(dst)}
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = direct_supertypes | indirect_supertypes

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_subtypes(dbp_type: str) -> set:
    subtypes = _get_subtype_mapping()[dbp_type]
    return subtypes | {et for t in subtypes for et in _get_equivalent_type_mapping()[t]}


def get_transitive_subtypes(dbp_type: str) -> set:
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        direct_subtypes = get_subtypes(dbp_type)
        indirect_subtypes = {tst for dst in direct_subtypes for tst in get_transitive_subtypes(dst)}
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = direct_subtypes | indirect_subtypes

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_independent_types(dbp_types: set) -> set:
    """Returns only types that are independent, i.e. there are no two types T, T' with T supertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


# BASIC GETTERS + INITIALIZERS
__TRANSITIVE_SUPERTYPE_MAPPING__ = defaultdict(set)
__TRANSITIVE_SUBTYPE_MAPPING__ = defaultdict(set)


__RESOURCE_TYPE_MAPPING__ = None


def _get_resource_type_mapping():
    global __RESOURCE_TYPE_MAPPING__
    if not __RESOURCE_TYPE_MAPPING__:
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', _create_resource_type_mapping)
    return __RESOURCE_TYPE_MAPPING__


def _create_resource_type_mapping():
    resource_type_mapping = defaultdict(list)
    for triple in rdfutil.parse_triples_from_file(util.get_data_file('files.dbpedia.instance_types')):
        if triple.pred == rdfutil.PREDICATE_TYPE:
            resource_type_mapping[triple.sub].append(triple.obj)
    for triple in rdfutil.parse_triples_from_file(util.get_data_file('files.dbpedia.transitive_instance_types')):
        if triple.pred == rdfutil.PREDICATE_TYPE:
            resource_type_mapping[triple.sub].append(triple.obj)

    index = resource_type_mapping.keys()
    columns = list({t for types in resource_type_mapping.values() for t in types})
    data = [[c in resource_type_mapping[i] for c in columns] for i in index]
    return pd.SparseDataFrame(data=data, index=index, columns=columns, dtype=bool)


__SUBTYPE_MAPPING__ = None
__SUPERTYPE_MAPPING__ = None
__EQUIVALENT_TYPE_MAPPING__ = None


def _get_subtype_mapping():
    if not __SUBTYPE_MAPPING__:
        _initialize_taxonomy()
    return __SUBTYPE_MAPPING__


def _get_supertype_mapping():
    if not __SUPERTYPE_MAPPING__:
        _initialize_taxonomy()
    return __SUPERTYPE_MAPPING__


def _get_equivalent_type_mapping():
    if not __EQUIVALENT_TYPE_MAPPING__:
        _initialize_taxonomy()
    return __EQUIVALENT_TYPE_MAPPING__


def _initialize_taxonomy():
    global __SUBTYPE_MAPPING__
    __SUBTYPE_MAPPING__ = defaultdict(set)
    global __SUPERTYPE_MAPPING__
    __SUPERTYPE_MAPPING__ = defaultdict(set)
    global __EQUIVALENT_TYPE_MAPPING__
    __EQUIVALENT_TYPE_MAPPING__ = defaultdict(set)
    for triple in rdfutil.parse_triples_from_file(util.get_data_file('files.dbpedia.taxonomy')):
        if triple.pred == rdfutil.PREDICATE_SUBCLASS_OF:
            child, parent = triple.sub, triple.obj
            __SUBTYPE_MAPPING__[parent].add(child)
            __SUPERTYPE_MAPPING__[child].add(parent)
        elif triple.pred == rdfutil.PREDICATE_EQUIVALENT_CLASS:
            type_a, type_b = triple.sub, triple.obj
            __EQUIVALENT_TYPE_MAPPING__[type_a].add(type_b)
            __EQUIVALENT_TYPE_MAPPING__[type_b].add(type_a)
