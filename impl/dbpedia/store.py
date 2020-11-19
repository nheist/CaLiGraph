"""Functionality to retrieve everything related to DBpedia resources, properties, and types."""

import util
import impl.util.rdf as rdf_util
from . import util as dbp_util
import impl.category.util as cat_util
import impl.list.util as list_util
from collections import defaultdict
import networkx as nx
from typing import Optional
from polyleven import levenshtein

# TODO: Replace caching with functools LRU cache
# DBpedia RESOURCES


def get_resources() -> set:
    """Return all resources that are neither files nor lists."""
    global __RESOURCES__
    if '__RESOURCES__' not in globals():
        __RESOURCES__ = {res for res in get_raw_resources() if not list_util.is_listpage(res) and not list_util.is_listspage(res) and not dbp_util.is_file_resource(res)}
    return __RESOURCES__


def get_raw_resources() -> set:
    """Return all resources in DBpedia."""
    global __RAW_RESOURCES__
    if '__RAW_RESOURCES__' not in globals():
        __RAW_RESOURCES__ = set(_get_label_mapping()) | set(get_resource_property_mapping())
    return __RAW_RESOURCES__


def is_possible_resource(obj: str) -> bool:
    """Return True, if the given object is a potential DBpedia resource (i.e. valid URI, no file, no category, ..)."""
    return dbp_util.is_dbp_resource(obj) and not dbp_util.is_file_resource(obj) and not cat_util.is_category(obj) and not list_util.is_listpage(obj) and not list_util.is_listspage(obj)


def get_label(dbp_object: str) -> str:
    """Return the label of a DBpedia resource or type."""
    global __RESOURCE_LABELS__
    if '__RESOURCE_LABELS__' not in globals():
        __RESOURCE_LABELS__ = dict(_get_label_mapping())
        __RESOURCE_LABELS__.update(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_LABEL))
    return __RESOURCE_LABELS__[dbp_object] if dbp_object in __RESOURCE_LABELS__ else dbp_util.object2name(dbp_object)


def get_object_for_label(label: str) -> str:
    """Return the object that fits the given label."""
    global __RESOURCE_INVERSE_LABELS__
    global __ONTOLOGY_INVERSE_LABELS__
    if '__RESOURCE_INVERSE_LABELS__' not in globals():
        __RESOURCE_INVERSE_LABELS__ = {v: k for k, v in _get_label_mapping().items()}
        ontology_labels = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_LABEL)
        __ONTOLOGY_INVERSE_LABELS__ = {v: k for k, v in ontology_labels.items()}
    if label in __ONTOLOGY_INVERSE_LABELS__:
        return __ONTOLOGY_INVERSE_LABELS__[label]
    if label in __RESOURCE_INVERSE_LABELS__:
        return __RESOURCE_INVERSE_LABELS__[label]
    return dbp_util.name2resource(label)


def _get_label_mapping() -> dict:
    global __RESOURCE_LABEL_MAPPING__
    if '__RESOURCE_LABEL_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.labels')], rdf_util.PREDICATE_LABEL)
        __RESOURCE_LABEL_MAPPING__ = util.load_or_create_cache('dbpedia_resource_labels', initializer)
    return __RESOURCE_LABEL_MAPPING__


def get_abstract(dbp_resource: str) -> str:
    """Return the Wikipedia abstract text of the given resource."""
    global __RESOURCE_ABSTRACTS__
    if '__RESOURCE_ABSTRACTS__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.long_abstracts')], rdf_util.PREDICATE_ABSTRACT)
        __RESOURCE_ABSTRACTS__ = defaultdict(str, util.load_or_create_cache('dbpedia_resource_abstracts', initializer))
    return __RESOURCE_ABSTRACTS__[dbp_resource]


def get_lexicalisations(dbp_resource: str) -> dict:
    """Return all lexicalisations (i.e. anchor texts) of the given resource."""
    global __RESOURCE_LEXICALISATIONS__
    if '__RESOURCE_LEXICALISATIONS__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_freq_dict_from_rdf([util.get_data_file('files.dbpedia.anchor_texts')], rdf_util.PREDICATE_ANCHOR_TEXT)
        __RESOURCE_LEXICALISATIONS__ = util.load_or_create_cache('dbpedia_resource_lexicalisations', initializer)
    return __RESOURCE_LEXICALISATIONS__[dbp_resource]


def get_inverse_lexicalisations(text: str) -> dict:
    """Return all resources that fit to the given lexicalisation."""
    global __RESOURCE_INVERSE_LEXICALISATIONS__
    if '__RESOURCE_INVERSE_LEXICALISATIONS__' not in globals():
        __RESOURCE_INVERSE_LEXICALISATIONS__ = defaultdict(dict, util.load_or_create_cache('dbpedia_resource_inverse_lexicalisations', _compute_inverse_lexicalisations))
    return __RESOURCE_INVERSE_LEXICALISATIONS__[text.lower()]


def _compute_inverse_lexicalisations():
    inverse_lexicalisation_dict = rdf_util.create_multi_val_freq_dict_from_rdf([util.get_data_file('files.dbpedia.anchor_texts')], rdf_util.PREDICATE_ANCHOR_TEXT, reverse_key=True)
    for lex, resources in inverse_lexicalisation_dict.items():
        for res in set(resources.keys()):
            redirect_res = resolve_redirect(res)
            if res != redirect_res:
                if redirect_res in inverse_lexicalisation_dict[lex]:
                    inverse_lexicalisation_dict[lex][redirect_res] += inverse_lexicalisation_dict[lex][res]
                else:
                    inverse_lexicalisation_dict[lex][redirect_res] = inverse_lexicalisation_dict[lex][res]
                del inverse_lexicalisation_dict[lex][res]
    return inverse_lexicalisation_dict


def get_types(dbp_resource: str) -> set:
    """Return all types in DBpedia."""
    return {t for t in _get_resource_type_mapping()[dbp_resource] if dbp_util.is_dbp_type(t)}


def get_transitive_types(dbp_resource: str) -> set:
    """Return a resource's types as well as the transitive closure of these types."""
    transitive_types = {tt for t in get_types(dbp_resource) for tt in get_transitive_supertype_closure(t)}
    return {t for t in transitive_types if dbp_util.is_dbp_type(t)}


def get_direct_resources_for_type(dbp_type: str) -> set:
    """Return all direct resources for a given DBpedia type (i.e. having dbp_type as the most specific type)."""
    global __TYPE_DIRECT_RESOURCE_MAPPING__
    if '__TYPE_DIRECT_RESOURCE_MAPPING__' not in globals():
        __TYPE_DIRECT_RESOURCE_MAPPING__ = defaultdict(set)
        for r, ts in _get_resource_type_mapping().items():
            if r != resolve_redirect(r):
                continue
            for t in get_independent_types(ts):
                __TYPE_DIRECT_RESOURCE_MAPPING__[t].add(r)
    return __TYPE_DIRECT_RESOURCE_MAPPING__[dbp_type]


def get_all_resources_for_type(dbp_type: str) -> set:
    """Return all resources for a given DBpedia type."""
    global __TYPE_RESOURCE_MAPPING__
    if '__TYPE_RESOURCE_MAPPING__' not in globals():
        __TYPE_RESOURCE_MAPPING__ = defaultdict(set)
        for r in _get_resource_type_mapping():
            if r != resolve_redirect(r):
                continue
            for t in get_transitive_types(r):
                __TYPE_RESOURCE_MAPPING__[t].add(r)
    return __TYPE_RESOURCE_MAPPING__[dbp_type]


def _get_resource_type_mapping() -> dict:
    global __RESOURCE_TYPE_MAPPING__
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.instance_types')], rdf_util.PREDICATE_TYPE)
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)
    return __RESOURCE_TYPE_MAPPING__


def get_properties(dbp_resource: str) -> dict:
    """Return all properties where `dbp_resource` is the subject."""
    return get_resource_property_mapping()[dbp_resource]


def get_inverse_properties(dbp_resource: str) -> dict:
    """Return all properties where `dbp_resource` is the object."""
    return get_inverse_resource_property_mapping()[dbp_resource]


def resolve_redirect(dbp_resource: str, visited=None) -> str:
    """Return the resource to which `dbp_resource` redirects (if any) or `dbp_resource` itself."""
    global __REDIRECTS__
    if '__REDIRECTS__' not in globals():
        initializer = lambda: rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.redirects')], rdf_util.PREDICATE_REDIRECTS)
        __REDIRECTS__ = util.load_or_create_cache('dbpedia_resource_redirects', initializer)

    if dbp_resource in __REDIRECTS__:
        visited = visited or set()
        if dbp_resource not in visited:
            return resolve_redirect(__REDIRECTS__[dbp_resource], visited | {dbp_resource})
    return dbp_resource


def resolve_spelling_redirect(dbp_resource: str) -> str:
    redirect = resolve_redirect(dbp_resource)
    if levenshtein(dbp_resource, redirect, 2) > 2:
        # return original resource if the redirect links to a completely different resource
        return dbp_resource
    return redirect


def get_disambiguation_mapping() -> dict:
    global __DISAMBIGUATIONS__
    if '__DISAMBIGUATIONS__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.disambiguations')], rdf_util.PREDICATE_DISAMBIGUATES)
        __DISAMBIGUATIONS__ = defaultdict(set, util.load_or_create_cache('dbpedia_resource_disambiguations', initializer))
    return __DISAMBIGUATIONS__


def get_statistics(dbp_resource: str) -> dict:
    """Return information about the types and properties of `dbp_resource`."""
    global __RESOURCE_STATISTICS__
    if '__RESOURCE_STATISTICS__' not in globals():
        __RESOURCE_STATISTICS__ = {}
    if dbp_resource not in __RESOURCE_STATISTICS__:
        __RESOURCE_STATISTICS__[dbp_resource] = {
            'types': get_transitive_types(dbp_resource),
            'properties': {(pred, val) for pred, values in get_properties(dbp_resource).items() for val in values},
        }
    return __RESOURCE_STATISTICS__[dbp_resource]


# DBpedia property


def get_resource_property_mapping() -> dict:
    """Return a mapping from DBpedia resources to a dict containing property-value assignments (containing facts of DBpedia)."""
    global __RESOURCE_PROPERTY_MAPPING__
    if '__RESOURCE_PROPERTY_MAPPING__' not in globals():
        property_files = [util.get_data_file('files.dbpedia.mappingbased_literals'), util.get_data_file('files.dbpedia.mappingbased_objects')]
        initializer = lambda: rdf_util.create_dict_from_rdf(property_files)
        __RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_resource_properties', initializer)
    return __RESOURCE_PROPERTY_MAPPING__


def get_inverse_resource_property_mapping() -> dict:
    """Return a mapping from DBpedia resources to a dict containing property-value assignments (containing inverted facts of DBpedia)."""
    global __INVERSE_RESOURCE_PROPERTY_MAPPING__
    if '__INVERSE_RESOURCE_PROPERTY_MAPPING__' not in globals():
        initializer = lambda: rdf_util.create_dict_from_rdf([util.get_data_file('files.dbpedia.mappingbased_objects')], reverse_key=True)
        __INVERSE_RESOURCE_PROPERTY_MAPPING__ = util.load_or_create_cache('dbpedia_inverse_resource_properties', initializer)
    return __INVERSE_RESOURCE_PROPERTY_MAPPING__


def get_domain(dbp_predicate: str) -> Optional[str]:
    """Return the domain of a given predicate."""
    global __PREDICATE_DOMAIN__
    if '__PREDICATE_DOMAIN__' not in globals():
        __PREDICATE_DOMAIN__ = defaultdict(lambda: None, rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DOMAIN))
    return __PREDICATE_DOMAIN__[dbp_predicate]


def get_range(dbp_predicate: str) -> Optional[str]:
    """Return the range of a given predicate."""
    global __PREDICATE_RANGE__
    if '__PREDICATE_RANGE__' not in globals():
        __PREDICATE_RANGE__ = defaultdict(lambda: None, rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_RANGE))
    return __PREDICATE_RANGE__[dbp_predicate]


def get_equivalent_predicates(dbp_predicate: str) -> set:
    """Return all equivalent predicates of a given predicate."""
    global __EQUIVALENT_PREDICATE__
    if '__EQUIVALENT_PREDICATE__' not in globals():
        __EQUIVALENT_PREDICATE__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_PROPERTY)
    return __EQUIVALENT_PREDICATE__[dbp_predicate]


def is_object_property(dbp_predicate: str) -> bool:
    """Return True, if the predicate always has a resource as object."""
    global __OBJECT_PROPERTY__
    if '__OBJECT_PROPERTY__' not in globals():
        __OBJECT_PROPERTY__ = defaultdict(lambda: False)
    if dbp_predicate not in __OBJECT_PROPERTY__:
        if get_range(dbp_predicate):
            __OBJECT_PROPERTY__[dbp_predicate] = dbp_util.is_dbp_type(get_range(dbp_predicate))
        else:
            for props in get_resource_property_mapping().values():
                if dbp_predicate in props:
                    __OBJECT_PROPERTY__[dbp_predicate] = dbp_util.is_dbp_resource(props[dbp_predicate].pop())
                    break
    return __OBJECT_PROPERTY__[dbp_predicate]


def is_functional(dbp_predicate: str) -> bool:
    """Return True, if the predicate is functional (i.e. a resource has at most one value for the given predicate)."""
    global __PREDICATE_FUNCTIONAL__
    if '__PREDICATE_FUNCTIONAL__' not in globals():
        __PREDICATE_FUNCTIONAL__ = defaultdict(bool, util.load_or_create_cache('dbpedia_functional_predicates', _create_functional_predicate_dict))
    return __PREDICATE_FUNCTIONAL__[dbp_predicate]


def _create_functional_predicate_dict():
    predicate_resources_count = {pred: 0 for pred in get_all_predicates()}
    predicate_nonfunctional_count = {pred: 0 for pred in get_all_predicates()}

    resource_property_mapping = get_resource_property_mapping()
    for r in resource_property_mapping:
        for pred in resource_property_mapping[r]:
            predicate_resources_count[pred] += 1
            if len(resource_property_mapping[r][pred]) > 1:
                predicate_nonfunctional_count[pred] += 1
    return {pred: (predicate_nonfunctional_count[pred] / predicate_resources_count[pred]) < .05 for pred in get_all_predicates()}


def get_all_predicates() -> set:
    """Return all predicates in DBpedia."""
    global __PREDICATES__
    if '__PREDICATES__' not in globals():
        __PREDICATES__ = {pred for props in get_resource_property_mapping().values() for pred in props}
    return __PREDICATES__


# DBpedia types


def get_all_types() -> set:
    """Return all types in DBpedia."""
    return set(_get_type_graph().nodes)


def get_types_by_name(name: str) -> set:
    """Return all types that fit the given name."""
    global __TYPE_LABELS__
    if '__TYPE_LABELS__' not in globals():
        __TYPE_LABELS__ = defaultdict(set)
        for t in get_all_types():
            __TYPE_LABELS__[get_label(t).lower().split()[-1]].add(t)
    return __TYPE_LABELS__[name.lower()]


def get_independent_types(dbp_types: set) -> set:
    """Return only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_supertypes(dbp_type: str) -> set:
    """Return all parent types of the given type."""
    type_graph = _get_type_graph()
    return set(type_graph.predecessors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_supertypes(dbp_type: str) -> set:
    """Return all transitive parent types of the given type."""
    global __TRANSITIVE_SUPERTYPE_MAPPING__
    if '__TRANSITIVE_SUPERTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUPERTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = nx.ancestors(type_graph, dbp_type) if dbp_type in type_graph else set()
    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_transitive_supertype_closure(dbp_type: str) -> set:
    """Return `dbp_type` itself and its transitive supertypes."""
    return {dbp_type} | get_transitive_supertypes(dbp_type)


def get_subtypes(dbp_type: str) -> set:
    """Return all child types of the given type."""
    type_graph = _get_type_graph()
    return set(type_graph.successors(dbp_type)) if dbp_type in type_graph else set()


def get_transitive_subtypes(dbp_type: str) -> set:
    """Return all transitive child types of the given type."""
    global __TRANSITIVE_SUBTYPE_MAPPING__
    if '__TRANSITIVE_SUBTYPE_MAPPING__' not in globals():
        __TRANSITIVE_SUBTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        type_graph = _get_type_graph()
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = nx.descendants(type_graph, dbp_type) if dbp_type in type_graph else set()
    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_transitive_subtype_closure(dbp_type: str) -> set:
    """Return `dbp_type` itself and its transitive subtypes."""
    return {dbp_type} | get_transitive_subtypes(dbp_type)


def get_equivalent_types(dbp_type: str) -> set:
    """Return the set of equivalent types to the given type (including itself)."""
    global __EQUIVALENT_TYPE_MAPPING__
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS, reflexive=True)
        # remove external types from equivalent mappings as they are prone to errors
        __EQUIVALENT_TYPE_MAPPING__ = defaultdict(set, {t: {et for et in __EQUIVALENT_TYPE_MAPPING__[t] if dbp_util.is_dbp_type(et) or et == rdf_util.CLASS_OWL_THING} for t in __EQUIVALENT_TYPE_MAPPING__ if dbp_util.is_dbp_type(t) or t == rdf_util.CLASS_OWL_THING})
    return {dbp_type} | __EQUIVALENT_TYPE_MAPPING__[dbp_type]


def are_equivalent_types(dbp_types: set) -> bool:
    """Return True, if types are equivalent."""
    return dbp_types == get_equivalent_types(list(dbp_types)[0])


def get_main_equivalence_type(dbp_type: str) -> str:
    """Return the main equivalence type (i.e. the most strongly linked type in the taxonomy) for a given type."""
    valid_types = get_main_equivalence_types() | {rdf_util.CLASS_OWL_THING}
    return get_equivalent_types(dbp_type).intersection(valid_types).pop()


def get_main_equivalence_types() -> set:
    global __MAIN_EQUIVALENCE_TYPES__
    if '__MAIN_EQUIVALENCE_TYPES__' not in globals():
        __MAIN_EQUIVALENCE_TYPES__ = rdf_util.create_set_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, None)
    return __MAIN_EQUIVALENCE_TYPES__


REMOVED_DISJOINTNESS_AXIOMS = [{'http://dbpedia.org/ontology/Agent', 'http://dbpedia.org/ontology/Place'}]
ADDED_DISJOINTNESS_AXIOMS = [{'http://dbpedia.org/ontology/Person', 'http://dbpedia.org/ontology/Place'}, {'http://dbpedia.org/ontology/Family', 'http://dbpedia.org/ontology/Place'}]
def get_disjoint_types(dbp_type: str) -> set:
    """Return all types that are disjoint with `dbp_type` (excluding the wrong disjointness Agent<->Place)."""
    global __DISJOINT_TYPE_MAPPING__
    if '__DISJOINT_TYPE_MAPPING__' not in globals():
        __DISJOINT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_DISJOINT_WITH, reflexive=True)
        # add/remove custom axioms
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {k: {v for v in values if {k, v} not in REMOVED_DISJOINTNESS_AXIOMS} for k, values in __DISJOINT_TYPE_MAPPING__.items()})
        for a, b in ADDED_DISJOINTNESS_AXIOMS:
            __DISJOINT_TYPE_MAPPING__[a].add(b)
            __DISJOINT_TYPE_MAPPING__[b].add(a)

        # completing the subtype of each type with the subtypes of its disjoint types
        __DISJOINT_TYPE_MAPPING__ = defaultdict(set, {t: {st for dt in disjoint_types for st in get_transitive_subtype_closure(dt)} for t, disjoint_types in __DISJOINT_TYPE_MAPPING__.items()})
    return __DISJOINT_TYPE_MAPPING__[dbp_type]


def get_type_depth(dbp_type: str) -> int:
    """Return the shortest way from `dbp_type` to the root of the type graph."""
    global __TYPE_DEPTH__
    if '__TYPE_DEPTH__' not in globals():
        type_graph = _get_type_graph()
        __TYPE_DEPTH__ = defaultdict(lambda: 1, nx.shortest_path_length(type_graph, source=rdf_util.CLASS_OWL_THING))
    return __TYPE_DEPTH__[dbp_type]


def get_type_frequency(dbp_type: str) -> float:
    """Return the amount of resources having `dbp_type` as type."""
    global __TYPE_FREQUENCY__
    if '__TYPE_FREQUENCY__' not in globals():
        __TYPE_FREQUENCY__ = defaultdict(int, util.load_or_create_cache('dbpedia_resource_type_frequency', _compute_type_frequency))
    return __TYPE_FREQUENCY__[dbp_type]


def _compute_type_frequency() -> dict:
    type_counts = rdf_util.create_count_dict(_get_resource_type_mapping().values())
    return {t: t_count / len(_get_resource_type_mapping()) for t, t_count in type_counts.items()}


def get_type_lexicalisations(lemma: str) -> dict:
    """Return the type lexicalisation score for a set of lemmas (i.e. the probabilities of types given `lemmas`)."""
    global __TYPE_LEXICALISATIONS__
    if '__TYPE_LEXICALISATIONS__' not in globals():
        __TYPE_LEXICALISATIONS__ = defaultdict(dict, util.load_cache('dbpedia_type_lexicalisations'))
    return __TYPE_LEXICALISATIONS__[lemma.lower()]


def _get_type_graph() -> nx.DiGraph:
    """Return the initialised graph of DBpedia types."""
    global __TYPE_GRAPH__
    if '__TYPE_GRAPH__' not in globals():
        subtype_mapping = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # add missing types (i.e. those, that do not have subclasses at all)
        all_types = rdf_util.create_set_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_TYPE, rdf_util.CLASS_OWL_CLASS)
        subtype_mapping.update({et: set() for t in all_types for et in get_equivalent_types(t) if et not in subtype_mapping})
        # completing subtypes with subtypes of equivalent types
        subtype_mapping = {t: {est for et in get_equivalent_types(t) for st in subtype_mapping[et] for est in get_equivalent_types(st)} for t in set(subtype_mapping)}
        # remove non-dbpedia types from ontology
        subtype_mapping = {t: {st for st in sts if dbp_util.is_dbp_type(st) or st == rdf_util.CLASS_OWL_THING} for t, sts in subtype_mapping.items() if dbp_util.is_dbp_type(t) or t == rdf_util.CLASS_OWL_THING}
        __TYPE_GRAPH__ = nx.DiGraph(incoming_graph_data=[(t, st) for t, sts in subtype_mapping.items() for st in sts])
    return __TYPE_GRAPH__
