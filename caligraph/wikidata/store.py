from typing import Optional
import util
import caligraph.util.rdf as rdf_util
import caligraph.dbpedia.store as dbp_store


NAMESPACE_WIKIDATA = 'http://www.wikidata.org/entity/'


def get_properties(wikidata_resource: str) -> set:
    global __WIKIDATA_RESOURCE_PROPERTIES__
    if '__WIKIDATA_RESOURCE_PROPERTIES__' not in globals():
        initializer = lambda: rdf_util.create_tuple_dict_from_rdf([util.get_data_file('files.wikidata.resource_properties')])
        __WIKIDATA_RESOURCE_PROPERTIES__ = util.load_or_create_cache('wikidata_resource_properties', initializer)

    return __WIKIDATA_RESOURCE_PROPERTIES__[wikidata_resource]


def dbp_property2wikidata(dbp_property: str) -> Optional[str]:
    global __DBP_WIKIDATA_PROPERTY_MAPPING__
    if '__DBP_WIKIDATA_PROPERTY_MAPPING__' not in globals():
        __DBP_WIKIDATA_PROPERTY_MAPPING__ = {}

    if dbp_property not in __DBP_WIKIDATA_PROPERTY_MAPPING__:
        wikidata_equivalents = [p for p in dbp_store.get_equivalent_properties(dbp_property) if p.startswith(NAMESPACE_WIKIDATA)]
        __DBP_WIKIDATA_PROPERTY_MAPPING__[dbp_property] = wikidata_equivalents[0] if wikidata_equivalents else None

    return __DBP_WIKIDATA_PROPERTY_MAPPING__[dbp_property]


def dbp_type2wikidata(dbp_type: str) -> Optional[str]:
    global __DBP_WIKIDATA_TYPE_MAPPING__
    if '__DBP_WIKIDATA_TYPE_MAPPING__' not in globals():
        __DBP_WIKIDATA_TYPE_MAPPING__ = {}

    if dbp_type not in __DBP_WIKIDATA_TYPE_MAPPING__:
        wikidata_equivalents = [t for t in dbp_store.get_equivalent_types(dbp_type) if t.startswith(NAMESPACE_WIKIDATA)]
        __DBP_WIKIDATA_TYPE_MAPPING__[dbp_type] = wikidata_equivalents[0] if wikidata_equivalents else None

    return __DBP_WIKIDATA_TYPE_MAPPING__[dbp_type]


def dbp_resource2wikidata(dbp_resource: str) -> Optional[str]:
    global __DBP_WIKIDATA_RESOURCE_MAPPING__
    if '__DBP_WIKIDATA_RESOURCE_MAPPING__' not in globals():
        __DBP_WIKIDATA_RESOURCE_MAPPING__ = util.load_or_create_cache('wikidata_dbpedia_resource_mapping', _create_dbp_wikidata_resource_mapping)

    return __DBP_WIKIDATA_RESOURCE_MAPPING__[dbp_resource] if dbp_resource in __DBP_WIKIDATA_RESOURCE_MAPPING__ else None


def _create_dbp_wikidata_resource_mapping() -> dict:
    dbp_wikidata_resource_mapping = {}
    for dbp_res in dbp_store.get_resources():
        for lang_link in dbp_store.get_interlanguage_links(dbp_res):
            if lang_link.startswith(NAMESPACE_WIKIDATA):
                dbp_wikidata_resource_mapping[dbp_res] = lang_link
                break
    return dbp_wikidata_resource_mapping
