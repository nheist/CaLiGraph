from typing import Optional
import util
import caligraph.util.rdf as rdf_util
import caligraph.dbpedia.store as dbp_store
import caligraph.dbpedia.util as dbp_util
from caligraph.util.enum_match import Match


__NAMESPACE_WIKIDATA__ = 'http://www.wikidata.org/entity/'
__NAMESPACE_WIKIDATA_PREDICATE__ = 'http://www.wikidata.org/prop/direct/'


def resource_has_type(dbp_resource: str, dbp_type: str) -> Match:
    wikidata_resource = _dbp_resource2wikidata(dbp_resource)
    if not wikidata_resource:
        return Match.MISSING

    actual_wikidata_types = _get_property_values(wikidata_resource, _dbp_predicate2wikidata(rdf_util.PREDICATE_TYPE))
    if not actual_wikidata_types:
        return Match.MISSING

    actual_dbp_types = {dbp_type for wikidata_type in actual_wikidata_types for dbp_type in _wikidata_type2dbp(wikidata_type)}
    transitive_actual_dbp_types = {tt for t in actual_dbp_types for tt in dbp_store.get_transitive_supertype_closure(t)}
    if dbp_type in transitive_actual_dbp_types:
        return Match.EXACT

    transitive_actual_dbp_subtypes = {st for t in actual_dbp_types for st in dbp_store.get_transitive_subtypes(t)}
    if dbp_type in transitive_actual_dbp_subtypes:
        return Match.PARTIAL

    return Match.NONE


def resource_has_property(dbp_resource: str, dbp_predicate: str, value: str) -> Match:
    wikidata_resource = _dbp_resource2wikidata(dbp_resource)
    if not wikidata_resource:
        return Match.MISSING

    wikidata_predicate = _dbp_predicate2wikidata(dbp_predicate)
    if not wikidata_predicate:
        return Match.MISSING

    actual_values = _get_property_values(wikidata_resource, wikidata_predicate)
    if not actual_values:
        return Match.MISSING

    wikidata_value = _dbp_resource2wikidata(value) if value.startswith(dbp_util.NAMESPACE_DBP_RESOURCE) else value
    return Match.EXACT if wikidata_value in actual_values else Match.NONE


def _get_property_values(wikidata_resource: str, wikidata_predicates: set) -> set:
    global __WIKIDATA_RESOURCE_PROPERTIES__
    if '__WIKIDATA_RESOURCE_PROPERTIES__' not in globals():
        valid_predicates = {wp for dbp_predicate in dbp_store.get_all_predicates() for wp in _dbp_predicate2wikidata(dbp_predicate)} | {rdf_util.PREDICATE_TYPE}
        initializer = lambda: rdf_util.create_dict_from_rdf([util.get_data_file('files.wikidata.resource_properties')], valid_predicates=valid_predicates)
        __WIKIDATA_RESOURCE_PROPERTIES__ = util.load_or_create_cache('wikidata_resource_properties', initializer)

    return {val for pred in wikidata_predicates for val in __WIKIDATA_RESOURCE_PROPERTIES__[wikidata_resource][pred]}


def _dbp_predicate2wikidata(dbp_predicate: str) -> set:
    global __DBP_WIKIDATA_PREDICATE_MAPPING__
    if '__DBP_WIKIDATA_PREDICATE_MAPPING__' not in globals():
        __DBP_WIKIDATA_PREDICATE_MAPPING__ = {}

    if dbp_predicate not in __DBP_WIKIDATA_PREDICATE_MAPPING__:
        wikidata_predicates = {_convert_to_predicate_namespace(p) for p in dbp_store.get_equivalent_predicates(dbp_predicate) if p.startswith(__NAMESPACE_WIKIDATA__)}
        __DBP_WIKIDATA_PREDICATE_MAPPING__[dbp_predicate] = wikidata_predicates

    return __DBP_WIKIDATA_PREDICATE_MAPPING__[dbp_predicate]


def _convert_to_predicate_namespace(wikidata_predicate: str) -> str:
    return __NAMESPACE_WIKIDATA_PREDICATE__ + wikidata_predicate[len(__NAMESPACE_WIKIDATA__):]


def _dbp_type2wikidata(dbp_type: str) -> set:
    global __DBP_WIKIDATA_TYPE_MAPPING__
    if '__DBP_WIKIDATA_TYPE_MAPPING__' not in globals():
        __DBP_WIKIDATA_TYPE_MAPPING__ = {}

    if dbp_type not in __DBP_WIKIDATA_TYPE_MAPPING__:
        __DBP_WIKIDATA_TYPE_MAPPING__[dbp_type] = {t for t in dbp_store.get_equivalent_types(dbp_type) if t.startswith(__NAMESPACE_WIKIDATA__)}

    return __DBP_WIKIDATA_TYPE_MAPPING__[dbp_type]


def _wikidata_type2dbp(wikidata_type: str) -> set:
    global __WIKIDATA_DBP_TYPE_MAPPING__
    if '__WIKIDATA_DBP_TYPE_MAPPING__' not in globals():
        __WIKIDATA_DBP_TYPE_MAPPING__ = {}

    if wikidata_type not in __WIKIDATA_DBP_TYPE_MAPPING__:
        __WIKIDATA_DBP_TYPE_MAPPING__[wikidata_type] = {t for t in dbp_store.get_equivalent_types(wikidata_type) if t.startswith(dbp_util.NAMESPACE_DBP_ONTOLOGY)}

    return __WIKIDATA_DBP_TYPE_MAPPING__[wikidata_type]


def _dbp_resource2wikidata(dbp_resource: str) -> Optional[str]:
    global __DBP_WIKIDATA_RESOURCE_MAPPING__
    if '__DBP_WIKIDATA_RESOURCE_MAPPING__' not in globals():
        __DBP_WIKIDATA_RESOURCE_MAPPING__ = util.load_or_create_cache('wikidata_dbpedia_resource_mapping', _create_dbp_wikidata_resource_mapping)

    return __DBP_WIKIDATA_RESOURCE_MAPPING__[dbp_resource] if dbp_resource in __DBP_WIKIDATA_RESOURCE_MAPPING__ else None


def _create_dbp_wikidata_resource_mapping() -> dict:
    dbp_wikidata_resource_mapping = {}
    for dbp_res in dbp_store.get_resources():
        for lang_link in dbp_store.get_interlanguage_links(dbp_res):
            if lang_link.startswith(__NAMESPACE_WIKIDATA__):
                dbp_wikidata_resource_mapping[dbp_res] = lang_link
                break
    return dbp_wikidata_resource_mapping
